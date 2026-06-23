"""Experimental entity-specific v3 refinement; never reconstructs a page mosaic."""

from __future__ import annotations

import cv2

from ..models import DetectedRegion, EntityEvidence, EntityType, TransformKind, TransformSpec
from ..utils import image_bytes_to_ndarray, ndarray_to_png_bytes


def refine_entity_input(region: DetectedRegion, working_crop_v2: bytes) -> EntityEvidence:
    """Create extractor-specific v3 evidence while retaining both source crops."""

    if region.entity_type == EntityType.IMAGE or not working_crop_v2:
        return EntityEvidence(
            region_id=region.region_id,
            entity_type=region.entity_type,
            source_crop_v1=region.original_crop,
            working_crop_v2=working_crop_v2,
            entity_input_v3=working_crop_v2 or region.original_crop,
        )

    profile = region.quality_assessment.profile if region.quality_assessment else None
    operations: list[TransformSpec] = []
    result = working_crop_v2
    if profile and profile.low_contrast is not None and profile.low_contrast >= 0.55:
        gray = image_bytes_to_ndarray(working_crop_v2, grayscale=True)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        result = ndarray_to_png_bytes(binary)
        operations.append(
            TransformSpec(
                kind=TransformKind.ADAPTIVE_BINARIZATION,
                parameters={"block_size": 31, "constant": 15},
                reason="text/table crop has weak local foreground separation",
            )
        )

    return EntityEvidence(
        region_id=region.region_id,
        entity_type=region.entity_type,
        source_crop_v1=region.original_crop,
        working_crop_v2=working_crop_v2,
        entity_input_v3=result,
        operations=operations,
    )
