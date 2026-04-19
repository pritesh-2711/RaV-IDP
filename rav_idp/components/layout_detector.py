"""Layout detection using Docling when available."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz

from ..config import DEFAULT_DPI
from ..models import DetectedRegion, EntityType, PageRecord
from ..utils import crop_image_bytes, docling_bbox_to_pixel_bbox

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency in tests
    DocumentConverter = None

_DOCLING_CONVERTER: DocumentConverter | None = None


def _normalize_docling_items(document: object, attr_name: str) -> Iterable[object]:
    return getattr(document, attr_name, []) or []


def _item_type(attr_name: str) -> EntityType:
    mapping = {
        "tables": EntityType.TABLE,
        "pictures": EntityType.IMAGE,
        "texts": EntityType.TEXT,
    }
    return mapping[attr_name]


def detect_layout(document_path: str | Path, page_records: list[PageRecord]) -> list[DetectedRegion]:
    """Detect layout elements and persist original crops."""

    global _DOCLING_CONVERTER
    if DocumentConverter is None:
        return []

    if _DOCLING_CONVERTER is None:
        _DOCLING_CONVERTER = DocumentConverter()
    result = _DOCLING_CONVERTER.convert(str(document_path))

    path = Path(document_path)
    is_pdf = path.suffix.lower() == ".pdf"
    with fitz.open(path) as doc:
        regions: list[DetectedRegion] = []
        counter = 0
        for attr_name in ("texts", "tables", "pictures"):
            for item in _normalize_docling_items(result.document, attr_name):
                prov = getattr(item, "prov", None) or []
                if not prov:
                    continue
                first_prov = prov[0]
                page_index = int(getattr(first_prov, "page_no", 1)) - 1
                if is_pdf:
                    page_height = float(doc[page_index].rect.height)
                else:
                    from ..utils import image_bytes_to_pil

                    _, image_height = image_bytes_to_pil(page_records[page_index].raw_image).size
                    page_height = float(image_height)
                pixel_bbox = docling_bbox_to_pixel_bbox(
                    first_prov.bbox,
                    page_height,
                    page_index,
                    dpi=DEFAULT_DPI,
                    scale=None if is_pdf else 1.0,
                )
                original_crop = crop_image_bytes(page_records[page_index].raw_image, pixel_bbox)
                raw_record = item.export_to_dict() if hasattr(item, "export_to_dict") else getattr(item, "__dict__", {})
                regions.append(
                    DetectedRegion(
                        region_id=f"{page_index}_{counter}",
                        entity_type=_item_type(attr_name),
                        bbox=pixel_bbox,
                        original_crop=original_crop,
                        processed_crop=original_crop,
                        raw_docling_record=raw_record,
                        page_index=page_index,
                    )
                )
                counter += 1

    return sorted(regions, key=lambda region: (region.page_index, region.bbox.y0, region.bbox.x0))
