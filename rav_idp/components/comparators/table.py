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
    """Estimate column count from the densest line using whitespace tokenisation.

    This is a best-effort heuristic. OCR output on table images does not
    reliably preserve multi-space column separators, so the result is often
    1 regardless of the actual column count. Prefer detected_col_count from
    a structure-recognition model when available.
    """
    best = 0
    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = [t for t in re.split(r"\t|\s{2,}", line) if t.strip()]
        best = max(best, len(tokens))
    return max(best, 1) if best else 0


def _has_visual_content(image_bytes: bytes, min_dark_ratio: float = 0.04) -> bool:
    """Return True if the binarized image has enough dark pixels to indicate real content."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = float(binary.sum()) / (255.0 * binary.size)
    return dark_ratio > min_dark_ratio


def _soft_count_match(predicted: int, reference_count: int, original_crop: bytes) -> float:
    """Ratio-based match between predicted count and a reference count.

    When both counts are zero, fall back to a visual-content check so that
    an empty prediction against a visually non-empty crop is not falsely
    treated as agreement.
    """
    if predicted == 0 and reference_count == 0:
        return 0.0 if _has_visual_content(original_crop) else 1.0
    if predicted == 0 or reference_count == 0:
        return 0.0
    return min(predicted, reference_count) / max(predicted, reference_count)


def _parse_ocr_to_signature(ocr_text: str) -> dict:
    """Parse OCR output into a structural signature for comparison.

    All non-empty lines are treated as cell content. Header extraction is
    intentionally omitted: OCR on a table image produces one text line per
    row, not one line per column, so splitting lines into header vs. data
    based on the predicted header count is self-referential and unreliable.
    The caller handles the absent header signal by absorbing the header weight
    into the cell accuracy term.
    """
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    return {
        "row_count": len(lines),
        "col_count": _ocr_col_count(ocr_text),
        "headers": [],
        "cells": lines,
    }


def compare_table(
    reconstruction: TableReconstruction,
    region: DetectedRegion,
    threshold: float,
    skip_visual: bool = False,
    detected_col_count: int | None = None,
) -> FidelityResult:
    """Compare reconstructed table output against the original crop.

    Parameters
    ----------
    skip_visual:
        When True, the SSIM visual component is skipped and the score equals
        f_struct directly. Use this when comparing a programmatically rendered
        grid against a real document photograph or scan — the two images are
        structurally dissimilar at the pixel level regardless of extraction
        quality, so SSIM provides no useful signal (e.g. standalone image-crop
        benchmarks such as PubTabNet evaluation).
    detected_col_count:
        Column count from a structure-detection model (e.g. TableTransformer).
        When provided, replaces the OCR-whitespace-derived column count for the
        col_match component. OCR does not reliably preserve multi-space column
        separators in table images, so a detector-supplied count is more
        accurate.
    """
    signature = reconstruction.structural_signature

    # --- visual component (skipped for image-crop benchmarks) ---
    visual_score: float | None = None
    if not skip_visual:
        rendered = binarize(reconstruction.rendered_image)
        original = binarize(region.original_crop)
        original_resized = cv2.resize(original, (rendered.shape[1], rendered.shape[0]))
        visual_score = float(ssim(rendered, original_resized, data_range=255))
        visual_score = max(0.0, min(1.0, visual_score))

    # --- structural component ---
    ocr_text = rapidocr_image_to_text(Image.open(io.BytesIO(region.original_crop)))
    ocr_signature = _parse_ocr_to_signature(ocr_text)

    # Row match: OCR line count is a reasonable row-count proxy.
    row_match = _soft_count_match(
        signature.get("row_count", 0),
        ocr_signature["row_count"],
        region.original_crop,
    )

    # Column match: prefer structure-detector count over OCR heuristic.
    col_reference = detected_col_count if detected_col_count is not None else ocr_signature["col_count"]
    col_match = _soft_count_match(
        signature.get("col_count", 0),
        col_reference,
        region.original_crop,
    )

    row_col_match = (row_match + col_match) / 2.0

    # Cell CER: extracted cell text vs. OCR reading of the original crop.
    cer_cells = compute_cer(
        " ".join(signature.get("cells", [])),
        " ".join(ocr_signature["cells"]),
    )

    # Structural score.
    # Original paper formula:
    #   f_struct = 0.2 * row_col + 0.3 * (1 - cer_headers) + 0.5 * (1 - cer_cells)
    # Header CER is dropped here because OCR cannot reliably produce per-column
    # header lines from a table image. The 0.3 header weight is absorbed into
    # the cell accuracy term, keeping weights summing to 1.0:
    #   f_struct = 0.2 * row_col + 0.8 * (1 - cer_cells)
    f_struct = max(0.0, min(1.0, 0.2 * row_col_match + 0.8 * (1.0 - cer_cells)))

    # --- combined score ---
    if skip_visual or visual_score is None:
        score = f_struct
    else:
        score = max(0.0, min(1.0, 0.4 * visual_score + 0.6 * f_struct))

    component_scores: dict = {
        "f_struct": round(f_struct, 4),
        "row_match": round(row_match, 4),
        "col_match": round(col_match, 4),
        "cer_cells": round(cer_cells, 4),
    }
    if visual_score is not None:
        component_scores["ssim"] = round(visual_score, 4)

    return FidelityResult(
        region_id=region.region_id,
        entity_type=EntityType.TABLE,
        fidelity_score=round(score, 4),
        passed_threshold=score >= threshold,
        threshold_used=threshold,
        component_scores=component_scores,
        extractor_name="primary",
    )
