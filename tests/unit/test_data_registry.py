from __future__ import annotations

from pathlib import Path

from rav_idp.data.downloader import DatasetDownloader
from rav_idp.data.registry import get_dataset_spec, list_datasets


def test_registry_contains_paper_datasets() -> None:
    keys = {dataset.key for dataset in list_datasets()}
    assert {"doclaynet", "pubtabnet", "funsd", "docvqa"}.issubset(keys)


def test_stage_external_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    downloader = DatasetDownloader(root=tmp_path / "data")
    result = downloader.stage_external("dociq", source)
    assert result.status == "staged"
    assert (downloader.dataset_dir("dociq") / "STAGED_FROM.txt").exists()
