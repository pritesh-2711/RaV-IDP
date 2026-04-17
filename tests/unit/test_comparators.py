from __future__ import annotations

from PIL import Image
import pytest

from rav_idp.components.comparators import table as table_comparator
from rav_idp.components.comparators.image import compare_image
from rav_idp.components.comparators.table import compare_table
from rav_idp.components.comparators.text import compare_text
from rav_idp.components.reconstructors.image import compute_phash
from rav_idp.models import BoundingBox, DetectedRegion, EntityType, ImageReconstruction, TableReconstruction, TextReconstruction
from rav_idp.utils import pil_to_png_bytes


def _bbox() -> BoundingBox:
    return BoundingBox(x0=0, y0=0, x1=100, y1=100, page=0)


def test_text_exact_match() -> None:
    result = compare_text(TextReconstruction(reocr_text="Hello world"), "Hello world", "0_0", 0.85)
    assert result.fidelity_score == 1.0


def test_text_both_empty() -> None:
    result = compare_text(TextReconstruction(reocr_text=""), "", "0_0", 0.85)
    assert result.fidelity_score == 1.0


def test_image_identical_crop() -> None:
    image_bytes = pil_to_png_bytes(Image.new("RGB", (32, 32), "white"))
    region = DetectedRegion(region_id="0_0", entity_type=EntityType.IMAGE, bbox=_bbox(), original_crop=image_bytes, raw_docling_record={}, page_index=0)
    reconstruction = ImageReconstruction(
        phash_hex=compute_phash(image_bytes),
        sharpness_crop=1.0,
        sharpness_original=1.0,
        caption_found=True,
    )
    result = compare_image(reconstruction, region, 0.70)
    assert result.component_scores["phash_similarity"] == 1.0


def test_table_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    image_bytes = pil_to_png_bytes(Image.new("RGB", (32, 32), "white"))
    region = DetectedRegion(region_id="0_0", entity_type=EntityType.TABLE, bbox=_bbox(), original_crop=image_bytes, raw_docling_record={}, page_index=0)
    reconstruction = TableReconstruction(
        rendered_image=image_bytes,
        structural_signature={"row_count": 0, "col_count": 0, "headers": [], "cells": []},
    )
    monkeypatch.setattr(table_comparator, "rapidocr_image_to_text", lambda _: "")
    result = compare_table(reconstruction, region, 0.75)
    assert 0.0 <= result.fidelity_score <= 1.0
