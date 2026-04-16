"""Region-level quality classification."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import DetectedRegion, QualityClass
from ..utils import image_bytes_to_ndarray


def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:20]:
        theta = line[0][1]
        angle = (theta * 180 / np.pi) - 90
        if -45 <= angle <= 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_region(region: DetectedRegion) -> DetectedRegion:
    """Classify a cropped region instead of the full page."""

    crop = region.original_crop
    if not crop:
        return region.model_copy(update={"quality_class": QualityClass.SCANNED_DEGRADED, "processed_crop": crop})

    gray = image_bytes_to_ndarray(crop, grayscale=True)
    flags: list[QualityClass] = []
    quality = QualityClass.CLEAN

    if region.entity_type == region.entity_type.IMAGE:
        quality = QualityClass.CLEAN
    else:
        skew = abs(_estimate_skew_angle(gray))
        sharpness = _sharpness(gray)
        if sharpness < 100 or skew > 1.5:
            quality = QualityClass.SCANNED_DEGRADED
        else:
            quality = QualityClass.SCANNED_CLEAN

    return region.model_copy(
        update={
            "quality_class": quality,
            "secondary_flags": flags,
            "processed_crop": crop,
        }
    )


def classify_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Classify all cropped regions."""

    return [classify_region(region) for region in regions]
