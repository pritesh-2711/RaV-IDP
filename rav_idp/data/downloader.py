"""Dataset acquisition and staging utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

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
        if not target_path.exists():
            urlretrieve(source.url, target_path)
            self._extract_if_archive(target_path, target_dir)
        return target_path

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
                archive.extractall(target_dir)
        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path) as archive:
                archive.extractall(target_dir)

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
