from __future__ import annotations

from PIL import Image

from rav_idp.components.reconstructors.image import reconstruct_image
from rav_idp.components.reconstructors.table import reconstruct_table
from rav_idp.components.reconstructors.text import reconstruct_text
from rav_idp.models import BoundingBox, DetectedRegion, EntityType, ExtractedEntity, ImageContent, TableContent, TextContent
from rav_idp.utils import pil_to_png_bytes


def _bbox() -> BoundingBox:
    return BoundingBox(x0=0, y0=0, x1=100, y1=100, page=0)


def test_table_reconstruct_empty_df() -> None:
    entity = ExtractedEntity(
        region_id="0_0",
        entity_type=EntityType.TABLE,
        extractor_name="primary",
        content=TableContent(
            dataframe_json='{"columns":["A","B","C"],"index":[],"data":[]}',
            markdown="",
            csv="A,B,C\n",
            headers=["A", "B", "C"],
            row_count=0,
            col_count=3,
        ),
    )
    region = DetectedRegion(region_id="0_0", entity_type=EntityType.TABLE, bbox=_bbox(), original_crop=b"", raw_docling_record={}, page_index=0)
    reconstruction = reconstruct_table(entity, region)
    assert reconstruction.content.rendered_image
    assert reconstruction.content.structural_signature["row_count"] == 0


def test_image_reconstruct_caption_found() -> None:
    image_bytes = pil_to_png_bytes(Image.new("RGB", (64, 64), "white"))
    region = DetectedRegion(region_id="0_0", entity_type=EntityType.IMAGE, bbox=_bbox(), original_crop=image_bytes, raw_docling_record={}, page_index=0)
    text_region = DetectedRegion(
        region_id="0_1",
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=105, x1=80, y1=125, page=0),
        original_crop=image_bytes,
        raw_docling_record={"text": "Figure 1. Caption"},
        page_index=0,
    )
    entity = ExtractedEntity(
        region_id="0_0",
        entity_type=EntityType.IMAGE,
        extractor_name="primary",
        content=ImageContent(crop_bytes=image_bytes, classification_label=None, classification_confidence=None),
    )
    reconstruction = reconstruct_image(entity, region, [region, text_region], caption_proximity_px=40)
    assert reconstruction.content.caption_found is True


def test_text_reconstruct_scanned() -> None:
    image_bytes = pil_to_png_bytes(Image.new("RGB", (80, 30), "white"))
    entity = ExtractedEntity(
        region_id="0_0",
        entity_type=EntityType.TEXT,
        extractor_name="primary",
        content=TextContent(text="Hello", urls=[]),
    )
    region = DetectedRegion(region_id="0_0", entity_type=EntityType.TEXT, bbox=_bbox(), original_crop=image_bytes, raw_docling_record={"text": "Hello"}, page_index=0)
    reconstruction = reconstruct_text(entity, region, is_native_pdf=False, document_path="dummy.png")
    assert reconstruction.content.reocr_text is not None
