from __future__ import annotations

from rav_idp.components.fallback_extractor import _parse_fallback_response
from rav_idp.models import BoundingBox, DetectedRegion, EntityType


def test_fallback_table_response_parsing() -> None:
    region = DetectedRegion(
        region_id="0_0",
        entity_type=EntityType.TABLE,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=0),
        original_crop=b"",
        raw_docling_record={},
        page_index=0,
    )
    entity = _parse_fallback_response({"headers": ["A"], "rows": [["1"]], "notes": []}, region)
    assert entity.extractor_name == "fallback"
    assert entity.content.row_count == 1


def test_fallback_text_response_parsing() -> None:
    region = DetectedRegion(
        region_id="0_1",
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=0),
        original_crop=b"",
        raw_docling_record={},
        page_index=0,
    )
    entity = _parse_fallback_response({"text": "Hello world"}, region)
    assert entity.content.text == "Hello world"
