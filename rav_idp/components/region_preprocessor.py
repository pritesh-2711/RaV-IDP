"""Region-level preprocessing after layout segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import DetectedRegion, EntityType, QualityClass
from ..utils import image_bytes_to_ndarray, ndarray_to_png_bytes


def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 250))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    center = (gray.shape[1] // 2, gray.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC)


def _binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_region(region: DetectedRegion) -> DetectedRegion:
    """Apply preprocessing to a region crop after layout separation."""

    crop = region.original_crop
    if not crop:
        return region.model_copy(update={"processed_crop": crop})
    if region.entity_type == EntityType.IMAGE:
        return region.model_copy(update={"processed_crop": crop})

    gray = image_bytes_to_ndarray(crop, grayscale=True)
    quality = region.quality_class or QualityClass.CLEAN

    if quality == QualityClass.SCANNED_DEGRADED:
        gray = _deskew(gray)
        gray = _binarize(gray)
    elif quality in {QualityClass.HANDWRITTEN, QualityClass.PHOTOGRAPHED}:
        gray = _clahe(gray)

    processed = ndarray_to_png_bytes(gray)
    return region.model_copy(update={"processed_crop": processed})


def preprocess_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Preprocess all regions after region-level classification."""

    return [preprocess_region(region) for region in regions]
