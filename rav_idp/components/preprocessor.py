"""Page pre-processing."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import PageRecord, QualityClass
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


def _process_page(page: PageRecord) -> bytes:
    if page.quality_class in {QualityClass.CLEAN, QualityClass.SCANNED_CLEAN}:
        return page.raw_image

    gray = image_bytes_to_ndarray(page.raw_image, grayscale=True)

    if page.quality_class == QualityClass.SCANNED_DEGRADED:
        gray = _deskew(gray)
        gray = _binarize(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif page.quality_class == QualityClass.PHOTOGRAPHED:
        gray = _clahe(gray)
    elif page.quality_class == QualityClass.OVERLAPPING:
        gray = cv2.fastNlMeansDenoising(gray)
    elif page.quality_class == QualityClass.HANDWRITTEN:
        gray = _clahe(gray)

    return ndarray_to_png_bytes(gray)


def preprocess_pages(page_records: list[PageRecord]) -> list[PageRecord]:
    """Update processed images while leaving raw images unchanged."""

    processed: list[PageRecord] = []
    for page in page_records:
        processed.append(page.model_copy(update={"processed_image": _process_page(page)}))
    return processed
