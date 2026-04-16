"""Render document pages without applying quality routing."""

from __future__ import annotations

from pathlib import Path

import fitz

from ..config import DEFAULT_DPI
from ..models import PageRecord, QualityClass
from ..utils import render_page_to_png


def render_document_pages(document_path: str | Path) -> list[PageRecord]:
    """Render input pages to raw images with no pre-classification assumptions."""

    path = Path(document_path)
    if path.suffix.lower() != ".pdf":
        image = path.read_bytes()
        return [
            PageRecord(
                page_index=0,
                quality_class=QualityClass.CLEAN,
                secondary_flags=[],
                raw_image=image,
                processed_image=image,
            )
        ]

    records: list[PageRecord] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            raw_image = render_page_to_png(page, dpi=DEFAULT_DPI)
            records.append(
                PageRecord(
                    page_index=page_index,
                    quality_class=QualityClass.CLEAN,
                    secondary_flags=[],
                    raw_image=raw_image,
                    processed_image=raw_image,
                )
            )
    return records
