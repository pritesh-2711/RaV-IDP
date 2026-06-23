"""Deterministic, read-only document image quality profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import fitz
import numpy as np
import pytesseract
from pytesseract import Output, TesseractError

from ..config import QualityProfilerSettings, get_settings
from ..models import (
    AcquisitionMode,
    DetectedRegion,
    InputKind,
    PageRecord,
    QualityAssessment,
    QualityProfile,
)


_UNIMPLEMENTED_METRICS = (
    "shadow",
    "noise",
    "perspective",
    "blockiness",
    "handwriting_likelihood",
    "overlap_likelihood",
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _linear_scale(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("Quality calibration upper bound must exceed lower bound.")
    return _clip01((value - low) / (high - low))


def _decode_grayscale(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    midpoint = float(weights.sum()) / 2.0
    return float(sorted_values[np.searchsorted(cumulative, midpoint, side="left")])


def _estimate_skew(gray: np.ndarray) -> tuple[float | None, float, int]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(12, width // 24),
        minLineLength=max(12, width // 8),
        maxLineGap=max(3, width // 100),
    )
    if lines is None:
        return None, 0.0, 0

    angles: list[float] = []
    lengths: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        while angle >= 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) <= 30:
            angles.append(angle)
            lengths.append(length)

    if not angles:
        return None, 0.0, 0

    angle_values = np.asarray(angles, dtype=np.float64)
    weight_values = np.asarray(lengths, dtype=np.float64)
    median_angle = _weighted_median(angle_values, weight_values)
    aligned_weight = float(weight_values[np.abs(angle_values - median_angle) <= 2.0].sum())
    concentration = aligned_weight / max(float(weight_values.sum()), 1.0)
    coverage = min(1.0, float(weight_values.sum()) / max(float(width * 2), 1.0))
    confidence = _clip01(concentration * coverage)
    return median_angle, confidence, len(angles)


def _background_variation(gray: np.ndarray) -> float:
    min_dimension = min(gray.shape)
    kernel_size = max(3, min(101, int(round(min_dimension / 12))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    return float(np.std(background.astype(np.float32)))


def _text_density(gray: np.ndarray) -> float:
    _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if count <= 1:
        return 0.0
    image_area = float(gray.size)
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    valid = component_areas[(component_areas >= 2) & (component_areas <= image_area * 0.05)]
    return _clip01(float(valid.sum()) / max(image_area, 1.0))


def _local_content_contrast(gray: np.ndarray) -> tuple[float, int]:
    """Measure contrast in tiles containing ink instead of letting white margins dominate."""

    height, width = gray.shape
    row_edges = np.linspace(0, height, 9, dtype=int)
    column_edges = np.linspace(0, width, 9, dtype=int)
    contrasts: list[float] = []
    for row in range(8):
        for column in range(8):
            tile = gray[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            if tile.size == 0 or float(np.mean(tile < 240)) < 0.02:
                continue
            percentile_5, percentile_95 = np.percentile(tile, [5, 95])
            contrasts.append(float(percentile_95 - percentile_5))
    if not contrasts:
        percentile_1, percentile_99 = np.percentile(gray, [1, 99])
        return float(percentile_99 - percentile_1), 0
    return float(np.median(contrasts)), len(contrasts)


def _estimate_orientation(
    gray: np.ndarray,
    config: QualityProfilerSettings,
) -> tuple[int | None, float | None, float | None, str | None]:
    """Return the correction rotation reported by Tesseract OSD."""

    try:
        result = pytesseract.image_to_osd(gray, output_type=Output.DICT)
    except (TesseractError, RuntimeError, ValueError) as error:
        return None, None, None, f"orientation_unavailable:{type(error).__name__}"

    try:
        rotation = int(result["rotate"]) % 360
        raw_confidence = max(0.0, float(result.get("orientation_conf", 0.0)))
    except (KeyError, TypeError, ValueError):
        return None, None, None, "orientation_unavailable:invalid_osd_result"
    confidence = _clip01(raw_confidence / config.orientation_confidence_reference)
    if confidence >= 0.5:
        return rotation, confidence, raw_confidence, None

    # OSD is often indecisive on sparse cards. Compare OCR evidence at all four
    # orientations, retaining only aggregate scores (never recognized content).
    scores: dict[int, float] = {}
    character_counts: dict[int, int] = {}
    for candidate in (0, 90, 180, 270):
        candidate_image = _rotate_for_measurement(gray, candidate)
        try:
            data = pytesseract.image_to_data(
                candidate_image,
                config="--psm 11",
                output_type=Output.DICT,
            )
        except (TesseractError, RuntimeError, ValueError):
            continue
        score = 0.0
        character_count = 0
        for text, confidence_value in zip(data.get("text", []), data.get("conf", [])):
            alphanumeric_count = sum(character.isalnum() for character in str(text))
            try:
                word_confidence = float(confidence_value)
            except (TypeError, ValueError):
                continue
            if alphanumeric_count and word_confidence >= 20.0:
                score += word_confidence * alphanumeric_count
                character_count += alphanumeric_count
        scores[candidate] = score
        character_counts[candidate] = character_count

    if not scores:
        return None, confidence, raw_confidence, "orientation_unavailable:ocr_vote_failed"
    ranked = sorted(scores, key=scores.get, reverse=True)
    best_rotation = ranked[0]
    best_score = scores[best_rotation]
    second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
    if character_counts[best_rotation] < 20 or best_score <= 0:
        return None, confidence, raw_confidence, "orientation_unavailable:insufficient_text"
    vote_confidence = _clip01((best_score - second_score) / best_score)
    if vote_confidence < 0.10:
        return None, vote_confidence, raw_confidence, "orientation_unavailable:ambiguous_ocr_vote"
    return best_rotation, vote_confidence, raw_confidence, "orientation_fallback:ocr_rotation_vote"


def _rotate_for_measurement(gray: np.ndarray, rotation: int | None) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


def _pdf_page_evidence(
    page: fitz.Page,
    metadata: dict[str, str],
    config: QualityProfilerSettings,
) -> tuple[InputKind, AcquisitionMode, dict[str, float | None], list[str]]:
    text_character_count = len("".join(page.get_text("text").split()))
    image_info = page.get_image_info(xrefs=True)
    page_area = max(float(page.rect.get_area()), 1.0)
    coverages: list[float] = []
    for image in image_info:
        try:
            intersection = fitz.Rect(image["bbox"]) & page.rect
            coverages.append(max(0.0, float(intersection.get_area()) / page_area))
        except (KeyError, TypeError, ValueError):
            continue
    maximum_image_coverage = max(coverages, default=0.0)
    full_page_raster = maximum_image_coverage >= config.full_page_raster_coverage
    sparse_text = text_character_count <= config.sparse_pdf_text_chars

    metadata_text = " ".join(
        str(metadata.get(key, "")).lower() for key in ("author", "creator", "producer")
    )
    scanner_markers = ("camscanner", "scanner", "scanned", "intsig")
    scanner_marker = next((marker for marker in scanner_markers if marker in metadata_text), None)
    warnings: list[str] = []

    if full_page_raster and sparse_text:
        input_kind = InputKind.RASTER_PDF
        acquisition_mode = AcquisitionMode.SCANNED if scanner_marker else AcquisitionMode.UNKNOWN
        warnings.append("pdf_scan_wrapper:full_page_raster_with_sparse_text")
        if scanner_marker:
            warnings.append(f"acquisition_inferred_from_metadata:{scanner_marker}")
    elif full_page_raster and text_character_count:
        input_kind = InputKind.NATIVE_PDF
        acquisition_mode = AcquisitionMode.MIXED
        warnings.append("pdf_mixed_content:full_page_raster_with_text_layer")
    elif text_character_count:
        input_kind = InputKind.NATIVE_PDF
        acquisition_mode = AcquisitionMode.DIGITAL
    else:
        input_kind = InputKind.RASTER_PDF
        acquisition_mode = AcquisitionMode.SCANNED if scanner_marker else AcquisitionMode.UNKNOWN
        if scanner_marker:
            warnings.append(f"acquisition_inferred_from_metadata:{scanner_marker}")

    measurements: dict[str, float | None] = {
        "pdf_text_character_count": float(text_character_count),
        "pdf_image_count": float(len(image_info)),
        "pdf_max_image_coverage": maximum_image_coverage,
    }
    return input_kind, acquisition_mode, measurements, warnings


def _derived_label(
    input_kind: InputKind,
    acquisition_mode: AcquisitionMode,
    severities: list[float | None],
) -> str:
    measured = [value for value in severities if value is not None]
    materially_degraded = bool(measured) and max(measured) >= 0.5
    if acquisition_mode == AcquisitionMode.PHOTOGRAPHED:
        return "photographed"
    if acquisition_mode == AcquisitionMode.MIXED:
        return "mixed"
    if acquisition_mode == AcquisitionMode.SCANNED:
        return "scanned-degraded" if materially_degraded else "scanned-clean"
    if acquisition_mode == AcquisitionMode.DIGITAL and input_kind == InputKind.NATIVE_PDF:
        return "digital-clean" if not materially_degraded else "unknown"
    return "unknown"


def _unavailable_assessment(
    *,
    scope: Literal["page", "region"],
    page_index: int,
    region_id: str | None,
    input_kind: InputKind,
    acquisition_mode: AcquisitionMode,
    config: QualityProfilerSettings,
    warning: str,
) -> QualityAssessment:
    return QualityAssessment(
        scope=scope,
        page_index=page_index,
        region_id=region_id,
        profile=QualityProfile(
            input_kind=input_kind,
            acquisition_mode=acquisition_mode,
            metric_version=config.metric_version,
            raw_measurements={},
            warnings=[warning, f"metrics_unavailable:{','.join(_UNIMPLEMENTED_METRICS)}"],
        ),
    )


def profile_image(
    image_bytes: bytes,
    *,
    scope: Literal["page", "region"],
    page_index: int,
    region_id: str | None = None,
    input_kind: InputKind = InputKind.UNKNOWN,
    acquisition_mode: AcquisitionMode = AcquisitionMode.UNKNOWN,
    config: QualityProfilerSettings | None = None,
    enable_orientation: bool = True,
) -> QualityAssessment:
    """Profile encoded image bytes without modifying them."""

    profiler_config = config or get_settings().quality_profiler
    gray = _decode_grayscale(image_bytes)
    if gray is None:
        return _unavailable_assessment(
            scope=scope,
            page_index=page_index,
            region_id=region_id,
            input_kind=input_kind,
            acquisition_mode=acquisition_mode,
            config=profiler_config,
            warning="image_decode_failed",
        )

    height, width = gray.shape
    if min(height, width) < profiler_config.min_dimension_px:
        return _unavailable_assessment(
            scope=scope,
            page_index=page_index,
            region_id=region_id,
            input_kind=input_kind,
            acquisition_mode=acquisition_mode,
            config=profiler_config,
            warning=f"image_too_small:{width}x{height}",
        )

    gray_float = gray.astype(np.float32)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    percentile_1, percentile_99 = np.percentile(gray_float, [1, 99])
    robust_contrast = float(percentile_99 - percentile_1)
    local_contrast, content_tile_count = _local_content_contrast(gray)
    brightness_mean = float(gray_float.mean())
    dark_fraction = float(np.mean(gray <= 32))
    bright_fraction = float(np.mean(gray >= 223))
    foreground_pixels = gray_float[gray_float < 250]
    foreground_level = (
        float(np.percentile(foreground_pixels, 10))
        if foreground_pixels.size >= max(8, int(gray.size * 0.001))
        else None
    )
    background_level = float(percentile_99)
    illumination_std = _background_variation(gray)
    text_density = _text_density(gray)
    rotation_required: int | None = None
    orientation_confidence: float | None = None
    orientation_confidence_raw: float | None = None
    orientation_warning: str | None = None
    if enable_orientation and scope == "page" and text_density > 0.001:
        (
            rotation_required,
            orientation_confidence,
            orientation_confidence_raw,
            orientation_warning,
        ) = _estimate_orientation(gray, profiler_config)
    measurement_gray = _rotate_for_measurement(gray, rotation_required)
    skew_angle, skew_confidence, skew_line_count = _estimate_skew(measurement_gray)

    blur = 1.0 - _linear_scale(
        sharpness,
        profiler_config.sharpness_blurred,
        profiler_config.sharpness_clean,
    )
    low_contrast = 1.0 - _linear_scale(
        local_contrast,
        profiler_config.contrast_low,
        profiler_config.contrast_clean,
    )
    underexposure = 1.0 - _linear_scale(
        background_level,
        profiler_config.underexposed_background,
        profiler_config.normal_background,
    )
    foreground_overexposure = (
        _linear_scale(
            foreground_level,
            profiler_config.normal_foreground,
            profiler_config.overexposed_foreground,
        )
        if foreground_level is not None
        else None
    )
    highlight_clipping = _linear_scale(bright_fraction, 0.60, 0.95) * low_contrast
    overexposure = (
        max(foreground_overexposure, highlight_clipping)
        if foreground_overexposure is not None
        else highlight_clipping
    )
    uneven_illumination = _clip01(illumination_std / profiler_config.illumination_std_severe)

    warnings = [f"metrics_unavailable:{','.join(_UNIMPLEMENTED_METRICS)}"]
    if orientation_warning is not None:
        warnings.append(orientation_warning)
    if rotation_required not in (None, 0) and (orientation_confidence or 0.0) < 0.5:
        warnings.append("orientation_low_confidence")
    if skew_angle is None or skew_confidence < profiler_config.minimum_skew_confidence:
        skew = None
        skew_angle_output = None
        warnings.append("skew_unavailable:no_reliable_lines")
    else:
        skew_angle_output = float(skew_angle)
        skew = _clip01(abs(skew_angle) / profiler_config.skew_severe_degrees)

    severities = [blur, low_contrast, underexposure, overexposure, uneven_illumination, skew]
    raw_measurements: dict[str, float | None] = {
        "sharpness_laplacian_variance": sharpness,
        "robust_contrast_p99_minus_p01": robust_contrast,
        "local_content_contrast_median": local_contrast,
        "content_contrast_tile_count": float(content_tile_count),
        "brightness_mean": brightness_mean,
        "background_level_p99": background_level,
        "foreground_level_p10": foreground_level,
        "dark_pixel_fraction": dark_fraction,
        "bright_pixel_fraction": bright_fraction,
        "highlight_clipping_score": highlight_clipping,
        "illumination_background_std": illumination_std,
        "skew_angle_degrees": skew_angle_output,
        "skew_confidence": skew_confidence,
        "skew_line_count": float(skew_line_count),
        "text_density": text_density,
        "orientation_confidence_raw": orientation_confidence_raw,
    }

    profile = QualityProfile(
        input_kind=input_kind,
        acquisition_mode=acquisition_mode,
        blur=_clip01(blur),
        low_contrast=_clip01(low_contrast),
        underexposure=_clip01(underexposure),
        overexposure=_clip01(overexposure) if overexposure is not None else None,
        uneven_illumination=_clip01(uneven_illumination),
        skew=skew,
        skew_angle_degrees=skew_angle_output,
        rotation_required_degrees=rotation_required,
        orientation_confidence=orientation_confidence,
        brightness_mean=brightness_mean,
        text_density=text_density,
        derived_label=_derived_label(input_kind, acquisition_mode, severities),
        metric_version=profiler_config.metric_version,
        raw_measurements=raw_measurements,
        warnings=warnings,
    )
    return QualityAssessment(
        scope=scope,
        page_index=page_index,
        region_id=region_id,
        profile=profile,
    )


def profile_pages(
    page_records: list[PageRecord],
    document_path: str | Path,
    config: QualityProfilerSettings | None = None,
) -> list[PageRecord]:
    """Attach page assessments while preserving all page image bytes."""

    path = Path(document_path)
    profiler_config = config or get_settings().quality_profiler
    evidence: dict[
        int,
        tuple[InputKind, AcquisitionMode, dict[str, float | None], list[str]],
    ] = {}
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            metadata = document.metadata or {}
            for page_index, page in enumerate(document):
                evidence[page_index] = _pdf_page_evidence(page, metadata, profiler_config)
    else:
        evidence[0] = (InputKind.IMAGE, AcquisitionMode.UNKNOWN, {}, [])

    profiled: list[PageRecord] = []
    for page in page_records:
        input_kind, acquisition_mode, pdf_measurements, evidence_warnings = evidence.get(
            page.page_index,
            (InputKind.UNKNOWN, AcquisitionMode.UNKNOWN, {}, []),
        )
        assessment = profile_image(
            page.raw_image,
            scope="page",
            page_index=page.page_index,
            input_kind=input_kind,
            acquisition_mode=acquisition_mode,
            config=profiler_config,
        )
        if pdf_measurements or evidence_warnings:
            profile = assessment.profile.model_copy(
                update={
                    "raw_measurements": {
                        **assessment.profile.raw_measurements,
                        **pdf_measurements,
                    },
                    "warnings": [*assessment.profile.warnings, *evidence_warnings],
                }
            )
            assessment = assessment.model_copy(update={"profile": profile})
        profiled.append(page.model_copy(update={"quality_assessment": assessment}))
    return profiled


def profile_region(
    region: DetectedRegion,
    page_assessment: QualityAssessment | None = None,
    config: QualityProfilerSettings | None = None,
) -> QualityAssessment:
    """Profile an immutable region crop, inheriting document evidence if known."""

    input_kind = InputKind.UNKNOWN
    acquisition_mode = AcquisitionMode.UNKNOWN
    if page_assessment is not None:
        input_kind = page_assessment.profile.input_kind
        acquisition_mode = page_assessment.profile.acquisition_mode
    assessment = profile_image(
        region.original_crop,
        scope="region",
        page_index=region.page_index,
        region_id=region.region_id,
        input_kind=input_kind,
        acquisition_mode=acquisition_mode,
        config=config,
    )
    profile = assessment.profile.model_copy(
        update={"warnings": [*assessment.profile.warnings, f"region_metric_context:{region.entity_type.value}"]}
    )
    return assessment.model_copy(
        update={"entity_type": region.entity_type, "bbox": region.bbox, "profile": profile}
    )
