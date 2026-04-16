from __future__ import annotations

from pathlib import Path

from rav_idp.components.quality_classifier import classify_document


def test_classify_non_pdf_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-data")
    pages = classify_document(image_path)
    assert len(pages) == 1
    assert pages[0].raw_image == b"png-data"
