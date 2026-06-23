from __future__ import annotations

from pathlib import Path

import pytest

from rav_idp.data.downloader import DatasetDownloader
from rav_idp.data.registry import DatasetSource, get_dataset_spec, list_datasets


def test_registry_contains_paper_datasets() -> None:
    keys = {dataset.key for dataset in list_datasets()}
    assert {"doclaynet", "pubtabnet", "funsd", "docvqa"}.issubset(keys)


def test_registry_contains_quality_and_restoration_datasets() -> None:
    keys = {dataset.key for dataset in list_datasets()}
    assert {
        "smartdoc_qa",
        "docunet",
        "uvdoc_benchmark",
        "doc3d",
        "dir300",
        "docreal",
        "dibco_series",
        "k_watermark",
        "mot_overlap",
    }.issubset(keys)
    smartdoc = get_dataset_spec("smartdoc_qa")
    assert smartdoc.sources[0].url.startswith("https://zenodo.org/")
    assert smartdoc.sources[0].checksum == "md5:643c5e54606626694b17ac5b8984baab"


def test_stage_external_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    downloader = DatasetDownloader(root=tmp_path / "data")
    result = downloader.stage_external("dociq", source)
    assert result.status == "staged"
    assert (downloader.dataset_dir("dociq") / "STAGED_FROM.txt").exists()


def test_download_verification_checks_size_and_checksum(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"trusted-dataset")
    source = DatasetSource(
        kind="http",
        url="https://example.test/payload.bin",
        expected_size_bytes=15,
        checksum="md5:35a9346f8174bc667ba7ae69d5bedc3c",
    )
    downloader = DatasetDownloader(root=tmp_path / "data")

    downloader._verify_download(payload, source)
    with pytest.raises(ValueError, match="Size mismatch"):
        downloader._verify_download(
            payload,
            source.model_copy(update={"expected_size_bytes": 14}),
        )


def test_archive_paths_cannot_escape_dataset_directory(tmp_path: Path) -> None:
    downloader = DatasetDownloader(root=tmp_path / "data")
    target = downloader.dataset_dir("docunet")
    target.mkdir(parents=True)

    downloader._validate_archive_paths(target, ["images/page.png"])
    with pytest.raises(ValueError, match="escapes target directory"):
        downloader._validate_archive_paths(target, ["../../outside.txt"])


def test_parallel_byte_ranges_cover_file_exactly() -> None:
    ranges = DatasetDownloader._byte_ranges(total_size=101, part_count=8)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100
    assert sum(end - start + 1 for start, end in ranges) == 101
    assert all(left[1] + 1 == right[0] for left, right in zip(ranges, ranges[1:]))
