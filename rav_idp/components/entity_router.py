"""Entity routing utilities."""

from __future__ import annotations

from ..models import DetectedRegion, EntityType


def route_entities(regions: list[DetectedRegion]) -> dict[str, list[DetectedRegion]]:
    """Partition regions by entity type."""

    routed = {entity_type.value: [] for entity_type in EntityType}
    for region in regions:
        routed[region.entity_type.value].append(region)
    return routed
