"""Text reconstructor."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from ...config import DEFAULT_DPI
from ...models import DetectedRegion, EntityType, ExtractedEntity, ReconstructedOutput, TextReconstruction


def reocr_crop(crop_bytes: bytes) -> str:
    if not crop_bytes:
        return ""
    image = Image.open(io.BytesIO(crop_bytes))
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def extract_pdf_text_stream(document_path: str | Path, region: DetectedRegion) -> str:
    factor = DEFAULT_DPI / 72
    with fitz.open(str(document_path)) as doc:
        page = doc[region.page_index]
        clip = fitz.Rect(
            region.bbox.x0 / factor,
            region.bbox.y0 / factor,
            region.bbox.x1 / factor,
            region.bbox.y1 / factor,
        )
        return page.get_text("text", clip=clip).strip()


def reconstruct_text(
    entity: ExtractedEntity,
    region: DetectedRegion,
    is_native_pdf: bool,
    document_path: str | Path,
) -> ReconstructedOutput:
    """Create an independent text reading for validation."""

    if entity.entity_type not in {EntityType.TEXT, EntityType.FORMULA, EntityType.URL}:
        raise ValueError("Text reconstruction requires a text-like entity.")
    content = TextReconstruction(
        reocr_text=extract_pdf_text_stream(document_path, region)
        if is_native_pdf
        else reocr_crop(region.processed_crop or region.original_crop)
    )
    return ReconstructedOutput(region_id=region.region_id, entity_type=EntityType.TEXT, content=content)
