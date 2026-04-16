"""Document quality classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from ..config import DEFAULT_DPI
from ..models import PageRecord, QualityClass
from ..utils import has_pdf_text_layer, image_bytes_to_ndarray, render_page_to_png


def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:30]:
        theta = line[0][1]
        angle = (theta * 180 / np.pi) - 90
        if -45 <= angle <= 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _aspect_quality_flags(gray: np.ndarray) -> set[QualityClass]:
    flags: set[QualityClass] = set()
    angle = abs(_estimate_skew_angle(gray))
    if angle > 1.5:
        flags.add(QualityClass.SCANNED_DEGRADED)
    if _sharpness(gray) < 100:
        flags.add(QualityClass.SCANNED_DEGRADED)
    return flags


def classify_document(document_path: str | Path) -> list[PageRecord]:
    """Render and classify document pages."""

    path = Path(document_path)
    if path.suffix.lower() != ".pdf":
        image = Path(document_path).read_bytes()
        return [
            PageRecord(
                page_index=0,
                quality_class=QualityClass.SCANNED_CLEAN,
                secondary_flags=[],
                raw_image=image,
                processed_image=image,
            )
        ]

    page_records: list[PageRecord] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            raw_image = render_page_to_png(page, dpi=DEFAULT_DPI)
            gray = image_bytes_to_ndarray(raw_image, grayscale=True)
            flags = _aspect_quality_flags(gray)

            if has_pdf_text_layer(page):
                quality_class = QualityClass.CLEAN
            elif QualityClass.SCANNED_DEGRADED in flags:
                quality_class = QualityClass.SCANNED_DEGRADED
            else:
                quality_class = QualityClass.SCANNED_CLEAN

            secondary_flags = sorted(flags - {quality_class}, key=lambda item: item.value)
            page_records.append(
                PageRecord(
                    page_index=page_index,
                    quality_class=quality_class,
                    secondary_flags=secondary_flags,
                    raw_image=raw_image,
                    processed_image=raw_image,
                )
            )

    return page_records
