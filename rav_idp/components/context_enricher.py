"""Context enrichment helpers."""

from __future__ import annotations

from ..models import BoundingBox, ContextRecord, DetectedRegion, EntityType, ExtractedEntity
from ..utils import centroid_distance

CAPTION_PREFIXES = ("Table", "Figure", "Fig.", "Chart", "Exhibit", "Appendix")
MAX_CAPTION_LENGTH = 300
CAPTION_PROXIMITY_PX = 60


def centroid(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def find_caption(region: DetectedRegion, all_regions: list[DetectedRegion]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for candidate in all_regions:
        if candidate.page_index != region.page_index or candidate.entity_type != EntityType.TEXT:
            continue
        text = str(candidate.raw_docling_record.get("text", ""))
        if not text or len(text) > MAX_CAPTION_LENGTH:
            continue
        if not any(text.startswith(prefix) for prefix in CAPTION_PREFIXES):
            continue
        vertical_gap = min(abs(candidate.bbox.y0 - region.bbox.y1), abs(region.bbox.y0 - candidate.bbox.y1))
        if vertical_gap <= CAPTION_PROXIMITY_PX:
            candidates.append((vertical_gap, text))
    return sorted(candidates)[0][1] if candidates else None


def nearest_neighbours(region: DetectedRegion, all_regions: list[DetectedRegion], k: int = 4) -> list[str]:
    distances: list[tuple[float, str]] = []
    for candidate in all_regions:
        if candidate.region_id == region.region_id or candidate.page_index != region.page_index:
            continue
        distances.append((centroid_distance(region.bbox, candidate.bbox), candidate.region_id))
    distances.sort(key=lambda item: item[0])
    return [region_id for _, region_id in distances[:k]]


def _text_regions_in_order(all_regions: list[DetectedRegion], page_index: int) -> list[DetectedRegion]:
    text_regions = [
        region
        for region in all_regions
        if region.page_index == page_index and region.entity_type == EntityType.TEXT
    ]
    return sorted(text_regions, key=lambda region: (region.bbox.y0, region.bbox.x0))


def enrich_context(
    entity: ExtractedEntity,
    region: DetectedRegion,
    all_regions: list[DetectedRegion],
    k_neighbours: int = 4,
) -> ContextRecord:
    """Attach caption, nearby text, and neighbouring region ids."""

    text_regions = _text_regions_in_order(all_regions, region.page_index)
    preceding = [
        str(candidate.raw_docling_record.get("text", ""))
        for candidate in text_regions
        if candidate.bbox.y0 < region.bbox.y0
    ][-2:]
    following = [
        str(candidate.raw_docling_record.get("text", ""))
        for candidate in text_regions
        if candidate.bbox.y0 > region.bbox.y1
    ][:2]

    return ContextRecord(
        region_id=entity.region_id,
        caption_text=find_caption(region, all_regions),
        preceding_text=preceding,
        following_text=following,
        neighbor_region_ids=nearest_neighbours(region, all_regions, k=k_neighbours),
    )
