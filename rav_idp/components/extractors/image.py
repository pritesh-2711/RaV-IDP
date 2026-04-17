"""Image extractor."""

from __future__ import annotations

from pathlib import Path

import fitz

from ...config import DEFAULT_DPI
from ...models import DetectedRegion, EntityType, ExtractedEntity, ImageContent


def extract_image(region: DetectedRegion, document_path: str | Path, scale: int = 2) -> ExtractedEntity:
    """Crop the image region from the source document."""

    path = Path(document_path)
    if path.suffix.lower() != ".pdf":
        crop_bytes = region.original_crop
    elif region.bbox.x1 <= region.bbox.x0 or region.bbox.y1 <= region.bbox.y0:
        crop_bytes = b""
    else:
        with fitz.open(str(path)) as doc:
            page = doc[region.page_index]
            factor = DEFAULT_DPI / 72
            clip = fitz.Rect(
                region.bbox.x0 / factor,
                region.bbox.y0 / factor,
                region.bbox.x1 / factor,
                region.bbox.y1 / factor,
            )
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            crop_bytes = pix.tobytes("png")

    classification = region.raw_docling_record.get("classification", {})
    confidence = classification.get("confidence")
    if confidence is not None and confidence < 0.40:
        label = None
        confidence = None
    else:
        label = classification.get("label")

    return ExtractedEntity(
        region_id=region.region_id,
        entity_type=EntityType.IMAGE,
        content=ImageContent(
            crop_bytes=crop_bytes,
            classification_label=label,
            classification_confidence=confidence,
        ),
        extractor_name="primary",
    )
