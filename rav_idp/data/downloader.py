"""Dataset acquisition and staging utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

from huggingface_hub import snapshot_download

from ..config import get_settings
from .registry import DATASET_REGISTRY, DatasetAccess, DatasetSource, DatasetSpec, get_dataset_spec


@dataclass(frozen=True)
class DownloadResult:
    dataset_key: str
    status: str
    message: str
    target_dir: Path
    downloaded_files: tuple[Path, ...] = ()


class DatasetDownloader:
    """Download or stage datasets into a stable local directory layout."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = (root or settings.data_root).resolve()
        self.raw_root = self.root / "raw"
        self.external_root = self.root / "external"
        self.manifest_root = self.root / "manifests"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.external_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, key: str) -> Path:
        return self.raw_root / key

    def stage_external(self, key: str, source_path: str | Path) -> DownloadResult:
        spec = get_dataset_spec(key)
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        target_dir = self.dataset_dir(key)
        target_dir.mkdir(parents=True, exist_ok=True)
        marker = target_dir / "STAGED_FROM.txt"
        marker.write_text(f"{source}\n", encoding="utf-8")
        self._write_manifest(spec, "staged", [source], extra={"source_path": str(source)})
        return DownloadResult(
            dataset_key=key,
            status="staged",
            message=f"Staged external dataset for {key} from {source}",
            target_dir=target_dir,
            downloaded_files=(source,),
        )

    def fetch(self, key: str) -> DownloadResult:
        spec = get_dataset_spec(key)
        target_dir = self.dataset_dir(key)
        target_dir.mkdir(parents=True, exist_ok=True)

        if spec.access in {DatasetAccess.MANUAL, DatasetAccess.CONTACT, DatasetAccess.RESTRICTED}:
            self._write_manifest(spec, "manual_required", [], extra={"access": spec.access.value})
            return DownloadResult(
                dataset_key=key,
                status="manual_required",
                message=f"{spec.display_name} requires {spec.access.value} acquisition.",
                target_dir=target_dir,
            )

        downloaded_files: list[Path] = []
        for source in spec.sources:
            if source.kind == "http":
                downloaded_files.append(self._download_http(source, target_dir))
            elif source.kind == "huggingface":
                downloaded_files.extend(self._download_huggingface(source, target_dir))
            elif source.kind == "manual":
                continue

        self._write_manifest(spec, "fetched", downloaded_files)
        return DownloadResult(
            dataset_key=key,
            status="fetched",
            message=f"Prepared dataset directory for {spec.display_name}",
            target_dir=target_dir,
            downloaded_files=tuple(downloaded_files),
        )

    def fetch_many(self, keys: list[str] | None = None) -> list[DownloadResult]:
        requested_keys = keys or list(DATASET_REGISTRY)
        return [self.fetch(key) for key in requested_keys]

    def _download_http(self, source: DatasetSource, target_dir: Path) -> Path:
        filename = source.filename or self._filename_from_url(source.url)
        target_path = target_dir / filename
        if target_path.exists():
            self._verify_download(target_path, source)
        elif source.download_parts > 1:
            self._download_http_parallel(source, target_path)
            self._extract_if_archive(target_path, target_dir)
        else:
            partial_path = target_path.with_name(f"{target_path.name}.part")
            urlretrieve(source.url, partial_path)
            self._verify_download(partial_path, source)
            partial_path.replace(target_path)
            self._extract_if_archive(target_path, target_dir)
        return target_path

    @staticmethod
    def _byte_ranges(total_size: int, part_count: int) -> list[tuple[int, int]]:
        if total_size <= 0 or part_count <= 0:
            raise ValueError("Download size and part count must be positive.")
        part_count = min(part_count, total_size)
        base_size, remainder = divmod(total_size, part_count)
        ranges: list[tuple[int, int]] = []
        start = 0
        for index in range(part_count):
            length = base_size + (1 if index < remainder else 0)
            end = start + length - 1
            ranges.append((start, end))
            start = end + 1
        return ranges

    def _download_http_parallel(self, source: DatasetSource, target_path: Path) -> None:
        if source.expected_size_bytes is None:
            raise ValueError("Parallel HTTP downloads require expected_size_bytes.")
        ranges = self._byte_ranges(source.expected_size_bytes, source.download_parts)
        part_paths = [
            target_path.with_name(f"{target_path.name}.part.{index:03d}")
            for index in range(len(ranges))
        ]

        groups: list[tuple[Path, int, int, list[Path]]] = []
        tasks: list[tuple[Path, int, int]] = []
        for index, ((start, end), part_path) in enumerate(zip(ranges, part_paths)):
            expected_length = end - start + 1
            current_length = part_path.stat().st_size if part_path.exists() else 0
            if current_length == expected_length:
                continue
            if current_length > expected_length:
                part_path.unlink()
                current_length = 0
            remaining_start = start + current_length
            remaining_size = end - remaining_start + 1
            subdivisions = self._byte_ranges(remaining_size, min(4, remaining_size))
            chunk_paths: list[Path] = []
            for chunk_index, (local_start, local_end) in enumerate(subdivisions):
                chunk_path = target_path.with_name(
                    f"{target_path.name}.part.{index:03d}.chunk.{chunk_index:02d}"
                )
                chunk_start = remaining_start + local_start
                chunk_end = remaining_start + local_end
                chunk_paths.append(chunk_path)
                tasks.append((chunk_path, chunk_start, chunk_end))
            groups.append((part_path, current_length, expected_length, chunk_paths))

        def download_chunk(task: tuple[Path, int, int]) -> None:
            chunk_path, start, end = task
            expected_length = end - start + 1
            current_length = chunk_path.stat().st_size if chunk_path.exists() else 0
            if current_length == expected_length:
                return
            if current_length > expected_length:
                chunk_path.unlink()
                current_length = 0
            request_start = start + current_length
            request = Request(
                source.url,
                headers={
                    "Range": f"bytes={request_start}-{end}",
                    "User-Agent": "RaV-IDP-DatasetDownloader/1.0",
                },
            )
            with urlopen(request) as response:
                if response.getcode() != 206:
                    raise ValueError(
                        f"Server did not honor byte range for {target_path.name}: "
                        f"HTTP {response.getcode()}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise ValueError(
                        f"Unexpected Content-Range for {target_path.name}: {content_range}"
                    )
                with chunk_path.open("ab") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            actual_length = chunk_path.stat().st_size
            if actual_length != expected_length:
                raise ValueError(
                    f"Incomplete range for {target_path.name}: "
                    f"expected {expected_length}, got {actual_length}"
                )

        if tasks:
            with ThreadPoolExecutor(max_workers=min(source.download_parts, len(tasks))) as executor:
                list(executor.map(download_chunk, tasks))

        for part_path, current_length, expected_length, chunk_paths in groups:
            rebuilding_path = part_path.with_name(f"{part_path.name}.rebuilding")
            with rebuilding_path.open("wb") as output:
                if current_length:
                    with part_path.open("rb") as existing:
                        remaining = current_length
                        while remaining:
                            chunk = existing.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError(
                                    f"Existing range shrank while assembling {target_path.name}"
                                )
                            output.write(chunk)
                            remaining -= len(chunk)
                for chunk_path in chunk_paths:
                    with chunk_path.open("rb") as chunk:
                        shutil.copyfileobj(chunk, output, length=1024 * 1024)
            actual_length = rebuilding_path.stat().st_size
            if actual_length != expected_length:
                raise ValueError(
                    f"Incomplete assembled part for {target_path.name}: "
                    f"expected {expected_length}, got {actual_length}"
                )
            rebuilding_path.replace(part_path)
            for chunk_path in chunk_paths:
                chunk_path.unlink()

        assembling_path = target_path.with_name(f"{target_path.name}.assembling")
        with assembling_path.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, output, length=1024 * 1024)
        self._verify_download(assembling_path, source)
        assembling_path.replace(target_path)
        for part_path in part_paths:
            part_path.unlink()
        for stale_chunk in target_path.parent.glob(f"{target_path.name}.part.*.chunk.*"):
            stale_chunk.unlink()
        for stale_rebuild in target_path.parent.glob(f"{target_path.name}.part.*.rebuilding"):
            stale_rebuild.unlink()
        legacy_partial = target_path.with_name(f"{target_path.name}.part")
        if legacy_partial.exists():
            legacy_partial.unlink()

    def _verify_download(self, file_path: Path, source: DatasetSource) -> None:
        if source.expected_size_bytes is not None:
            actual_size = file_path.stat().st_size
            if actual_size != source.expected_size_bytes:
                raise ValueError(
                    f"Size mismatch for {file_path.name}: expected "
                    f"{source.expected_size_bytes}, got {actual_size}"
                )
        if source.checksum:
            algorithm, separator, expected = source.checksum.partition(":")
            if not separator or algorithm not in hashlib.algorithms_available:
                raise ValueError(f"Unsupported checksum specification: {source.checksum}")
            digest = hashlib.new(algorithm)
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual.lower() != expected.lower():
                raise ValueError(
                    f"Checksum mismatch for {file_path.name}: expected {expected}, got {actual}"
                )

    def _filename_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        return name or "download.bin"

    def _repo_id_from_hf_url(self, url: str) -> str:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "datasets":
            return "/".join(parts[1:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        raise ValueError(f"Could not parse Hugging Face repo id from URL: {url}")

    def _download_huggingface(self, source: DatasetSource, target_dir: Path) -> list[Path]:
        repo_id = self._repo_id_from_hf_url(source.url)
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            allow_patterns=source.allow_patterns or None,
            ignore_patterns=source.ignore_patterns or None,
        )
        snapshot_dir = Path(snapshot_path)
        return [snapshot_dir]

    def _extract_if_archive(self, file_path: Path, target_dir: Path) -> None:
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path) as archive:
                self._validate_archive_paths(
                    target_dir,
                    [member.filename for member in archive.infolist()],
                )
                archive.extractall(target_dir)
        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path) as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise ValueError(f"Archive contains links and will not be extracted: {file_path}")
                self._validate_archive_paths(target_dir, [member.name for member in members])
                archive.extractall(target_dir)

    def _validate_archive_paths(self, target_dir: Path, member_names: list[str]) -> None:
        root = target_dir.resolve()
        for member_name in member_names:
            destination = (root / member_name).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Archive member escapes target directory: {member_name}")

    def _write_manifest(self, spec: DatasetSpec, status: str, files: list[Path], extra: dict | None = None) -> None:
        payload = {
            "dataset_key": spec.key,
            "display_name": spec.display_name,
            "stage": spec.stage,
            "status": status,
            "files": [str(path) for path in files],
            "expected_artifacts": spec.expected_artifacts,
            "sources": [source.model_dump() for source in spec.sources],
        }
        if extra:
            payload.update(extra)
        manifest_path = self.manifest_root / f"{spec.key}.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
