"""Spatial region filtering utilities."""

from __future__ import annotations

from ..models import BoundingBox, DetectedRegion, EntityType

_CONTAINMENT_TYPES = {EntityType.IMAGE, EntityType.TABLE}
_TEXT_TYPES = {EntityType.TEXT, EntityType.FORMULA, EntityType.URL}


def _containment_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of inner bbox area that overlaps with outer bbox."""
    if inner.page != outer.page:
        return 0.0
    ix0 = max(inner.x0, outer.x0)
    iy0 = max(inner.y0, outer.y0)
    ix1 = min(inner.x1, outer.x1)
    iy1 = min(inner.y1, outer.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    overlap = (ix1 - ix0) * (iy1 - iy0)
    inner_area = (inner.x1 - inner.x0) * (inner.y1 - inner.y0)
    if inner_area <= 0:
        return 0.0
    return overlap / inner_area


def suppress_text_inside_images(
    regions: list[DetectedRegion],
    containment_threshold: float = 0.85,
) -> tuple[list[DetectedRegion], list[str]]:
    """Remove text/formula/url regions that are substantially contained within
    an image or table region on the same page.

    Returns the filtered region list and a list of suppressed region_ids.
    The containment_threshold is the minimum fraction of the text region's area
    that must overlap the image region before it is suppressed.
    """
    container_regions = [r for r in regions if r.entity_type in _CONTAINMENT_TYPES]
    suppressed_ids: list[str] = []
    kept: list[DetectedRegion] = []

    for region in regions:
        if region.entity_type not in _TEXT_TYPES:
            kept.append(region)
            continue
        contained = any(
            _containment_ratio(region.bbox, container.bbox) >= containment_threshold
            for container in container_regions
            if container.region_id != region.region_id
        )
        if contained:
            suppressed_ids.append(region.region_id)
        else:
            kept.append(region)

    return kept, suppressed_ids
