"""Table comparator."""

from __future__ import annotations

import io
import re

import cv2
import numpy as np
from Levenshtein import distance as lev_distance
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from ...models import DetectedRegion, EntityType, FidelityResult, TableReconstruction
from ...utils import rapidocr_image_to_text


def binarize(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((1, 1), dtype=np.uint8)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def compute_cer(hypothesis: str, reference: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(1.0, lev_distance(hypothesis, reference) / len(reference))



def _ocr_row_count(ocr_text: str) -> int:
    """Count non-empty lines in OCR output as an independent row-count proxy."""
    return sum(1 for line in ocr_text.splitlines() if line.strip())


def _ocr_col_count(ocr_text: str) -> int:
    """Estimate column count from the first non-empty OCR line using whitespace tokenisation."""
    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = [t for t in re.split(r"\t|\s{2,}", line) if t.strip()]
        return max(len(tokens), 1)
    return 0


def _has_visual_content(image_bytes: bytes, min_dark_ratio: float = 0.04) -> bool:
    """Return True if the binarized image has enough dark pixels to indicate real content."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = float(binary.sum()) / (255.0 * binary.size)
    return dark_ratio > min_dark_ratio


def _soft_count_match(predicted: int, ocr_estimate: int, original_crop: bytes) -> float:
    """Ratio-based match using OCR-derived row count.

    When OCR returns 0 (e.g. image too small for Tesseract), fall back to a
    visual-content check so that an empty prediction against a visually
    non-empty crop is not falsely treated as agreement.
    """
    if predicted == 0 and ocr_estimate == 0:
        # OCR is silent; use pixel content as tiebreaker.
        return 0.0 if _has_visual_content(original_crop) else 1.0
    if predicted == 0 or ocr_estimate == 0:
        return 0.0
    return min(predicted, ocr_estimate) / max(predicted, ocr_estimate)


def _parse_ocr_to_signature(ocr_text: str, reference_signature: dict) -> dict:
    """Parse OCR output into a comparable signature.

    Row and column counts are derived independently from the OCR text so
    that the structural match score is not self-referential against the
    predicted signature.
    """
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    # Use first line as header text; remaining lines as cell text.
    # The number of header lines is taken from the prediction only for
    # content alignment, not for count comparison.
    n_pred_headers = len(reference_signature.get("headers", []))
    header_split = max(n_pred_headers, 1) if lines else 0
    headers = lines[:header_split]
    cells = lines[header_split:]
    return {
        "row_count": _ocr_row_count(ocr_text),    # independent of prediction
        "col_count": _ocr_col_count(ocr_text),    # independent of prediction
        "headers": headers,
        "cells": cells,
    }


def compare_table(reconstruction: TableReconstruction, region: DetectedRegion, threshold: float) -> FidelityResult:
    """Compare reconstructed table output against the original crop."""

    rendered = binarize(reconstruction.rendered_image)
    original = binarize(region.original_crop)
    original_resized = cv2.resize(original, (rendered.shape[1], rendered.shape[0]))
    visual_score = float(ssim(rendered, original_resized, data_range=255))
    visual_score = max(0.0, min(1.0, visual_score))

    ocr_text = rapidocr_image_to_text(Image.open(io.BytesIO(region.original_crop)))
    ocr_signature = _parse_ocr_to_signature(ocr_text, reconstruction.structural_signature)
    signature = reconstruction.structural_signature

    # Soft ratio match against OCR-derived counts (no longer self-referential).
    row_match = _soft_count_match(signature.get("row_count", 0), ocr_signature.get("row_count", 0), region.original_crop)
    col_match = _soft_count_match(signature.get("col_count", 0), ocr_signature.get("col_count", 0), region.original_crop)
    row_col_match = (row_match + col_match) / 2.0

    cer_headers = compute_cer(" ".join(signature.get("headers", [])), " ".join(ocr_signature.get("headers", [])))
    cer_cells = compute_cer(" ".join(signature.get("cells", [])), " ".join(ocr_signature.get("cells", [])))

    f_struct = 0.2 * row_col_match + 0.3 * (1 - cer_headers) + 0.5 * (1 - cer_cells)
    f_struct = max(0.0, min(1.0, f_struct))
    score = max(0.0, min(1.0, 0.4 * visual_score + 0.6 * f_struct))

    return FidelityResult(
        region_id=region.region_id,
        entity_type=EntityType.TABLE,
        fidelity_score=round(score, 4),
        passed_threshold=score >= threshold,
        threshold_used=threshold,
        component_scores={
            "ssim": round(visual_score, 4),
            "f_struct": round(f_struct, 4),
            "row_match": round(row_match, 4),
            "col_match": round(col_match, 4),
            "cer_headers": round(cer_headers, 4),
            "cer_cells": round(cer_cells, 4),
        },
        extractor_name="primary",
    )
