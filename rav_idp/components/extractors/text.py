"""Text extractor."""

from __future__ import annotations

import re

from ...models import DetectedRegion, EntityType, ExtractedEntity, TextContent

URL_PATTERN = re.compile(
    r'\b((?:https?://|www\.)[^\s<>()\[\]{}"\']+)',
    re.IGNORECASE | re.VERBOSE,
)


def extract_text(region: DetectedRegion) -> ExtractedEntity:
    """Extract text and URLs from a docling text-like record."""

    text = str(region.raw_docling_record.get("text", ""))
    urls = sorted(set(URL_PATTERN.findall(text)))
    return ExtractedEntity(
        region_id=region.region_id,
        entity_type=region.entity_type if region.entity_type in {EntityType.TEXT, EntityType.FORMULA, EntityType.URL} else EntityType.TEXT,
        content=TextContent(text=text, urls=urls),
        extractor_name="primary",
    )
