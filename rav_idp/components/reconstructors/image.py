"""Image reconstructor."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from ...models import DetectedRegion, EntityType, ExtractedEntity, ImageReconstruction, ReconstructedOutput


def compute_phash(image_bytes: bytes) -> str:
    if not image_bytes:
        return "0" * 16
    import imagehash

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return str(imagehash.phash(image))


def compute_sharpness(image_bytes: bytes) -> float:
    if not image_bytes:
        return 0.0
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def check_caption_adjacency(region: DetectedRegion, all_regions: list[DetectedRegion], proximity_px: int) -> bool:
    for candidate in all_regions:
        if candidate.page_index != region.page_index or candidate.entity_type != EntityType.TEXT:
            continue
        vertical_gap = min(abs(candidate.bbox.y0 - region.bbox.y1), abs(region.bbox.y0 - candidate.bbox.y1))
        if vertical_gap <= proximity_px:
            return True
    return False


def reconstruct_image(
    entity: ExtractedEntity,
    region: DetectedRegion,
    all_regions: list[DetectedRegion],
    caption_proximity_px: int = 60,
) -> ReconstructedOutput:
    """Compute image reconstruction metrics."""

    if entity.entity_type != EntityType.IMAGE:
        raise ValueError("Image reconstruction requires an image entity.")
    content = ImageReconstruction(
        phash_hex=compute_phash(entity.content.crop_bytes),
        sharpness_crop=compute_sharpness(entity.content.crop_bytes),
        sharpness_original=compute_sharpness(region.original_crop),
        caption_found=check_caption_adjacency(region, all_regions, caption_proximity_px),
    )
    return ReconstructedOutput(region_id=region.region_id, entity_type=EntityType.IMAGE, content=content)
