"""Content-preservation checks for experimental page restoration."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from ..models import CoordinateMapping, RestorationIntegrity
from ..utils import image_bytes_to_ndarray


def map_points(
    points: list[tuple[float, float]],
    mapping: CoordinateMapping,
) -> list[tuple[float, float]]:
    matrix = np.asarray(mapping.matrix, dtype=np.float64).reshape(3, 3)
    homogeneous = np.asarray([[x, y, 1.0] for x, y in points], dtype=np.float64).T
    projected = matrix @ homogeneous
    projected /= np.maximum(np.abs(projected[2:3]), 1e-12)
    return [(float(x), float(y)) for x, y in projected[:2].T]


def evaluate_restoration_integrity(
    source_page_v1: bytes,
    working_page_v2: bytes,
    mapping: CoordinateMapping,
    *,
    page_index: int,
    threshold: float,
) -> RestorationIntegrity:
    """Compare v2 with v1 after mapping v2 pixels back into source coordinates."""

    source = image_bytes_to_ndarray(source_page_v1, grayscale=True)
    restored = image_bytes_to_ndarray(working_page_v2, grayscale=True)
    if source is None or restored is None:
        raise ValueError("Unable to decode evidence for integrity evaluation.")

    matrix = np.asarray(mapping.matrix, dtype=np.float64).reshape(3, 3)
    aligned = cv2.warpPerspective(
        restored,
        np.linalg.inv(matrix),
        (mapping.source_width, mapping.source_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    source_small = cv2.resize(source, (512, max(1, round(512 * source.shape[0] / source.shape[1]))))
    aligned_small = cv2.resize(aligned, (source_small.shape[1], source_small.shape[0]))
    similarity = float(structural_similarity(source_small, aligned_small, data_range=255))
    similarity = float(np.clip(similarity, 0.0, 1.0))

    source_foreground = max(float(np.mean(source < 245)), 1e-6)
    restored_foreground = float(np.mean(aligned < 245))
    foreground_retention = restored_foreground / source_foreground

    source_edges = cv2.Canny(source, 50, 150) > 0
    restored_edges = cv2.Canny(aligned, 50, 150) > 0
    edge_retention = float(np.count_nonzero(source_edges & restored_edges)) / max(
        float(np.count_nonzero(source_edges)),
        1.0,
    )

    warnings: list[str] = []
    if not 0.50 <= foreground_retention <= 2.0:
        warnings.append("foreground_retention_outside_safety_band")
    if edge_retention < 0.35:
        warnings.append("edge_retention_below_safety_band")
    passed = (
        similarity >= threshold
        and 0.50 <= foreground_retention <= 2.0
        and edge_retention >= 0.35
    )
    return RestorationIntegrity(
        page_index=page_index,
        passed=passed,
        threshold=threshold,
        structural_similarity=similarity,
        foreground_retention=foreground_retention,
        edge_retention=edge_retention,
        warnings=warnings,
    )
