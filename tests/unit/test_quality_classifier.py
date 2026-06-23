from __future__ import annotations

import json
from pathlib import Path

import cv2
import fitz
import numpy as np

from rav_idp.components.quality_classifier import classify_document
from rav_idp.components.page_renderer import render_document_pages
from rav_idp.components.quality_profiler import profile_image, profile_pages, profile_region
from rav_idp.config import QualityProfilerSettings
from rav_idp.inspection import VisualArtifactRecorder
from rav_idp.models import (
    AcquisitionMode,
    BoundingBox,
    DetectedRegion,
    EntityType,
    InputKind,
    PageRecord,
    QualityClass,
)


def _encode(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _printed_page() -> np.ndarray:
    image = np.full((420, 640), 255, dtype=np.uint8)
    for index, text in enumerate(
        (
            "QUALITY ASSESSMENT",
            "The quick brown fox jumps over the lazy dog.",
            "Document restoration must preserve source evidence.",
            "0123456789  ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
    ):
        cv2.putText(
            image,
            text,
            (30, 70 + index * 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            0,
            2,
            cv2.LINE_AA,
        )
    cv2.line(image, (30, 365), (600, 365), 0, 2)
    return image


def _profile(image: np.ndarray):
    return profile_image(
        _encode(image),
        scope="page",
        page_index=0,
        input_kind=InputKind.IMAGE,
        acquisition_mode=AcquisitionMode.UNKNOWN,
    ).profile


def test_classify_non_pdf_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-data")
    pages = classify_document(image_path)
    assert len(pages) == 1
    assert pages[0].raw_image == b"png-data"
    assert pages[0].quality_assessment is not None
    assert "image_decode_failed" in pages[0].quality_assessment.profile.warnings


def test_profiler_is_deterministic_and_does_not_mutate_input() -> None:
    image_bytes = _encode(_printed_page())
    original = bytes(image_bytes)

    first = profile_image(image_bytes, scope="page", page_index=0)
    second = profile_image(image_bytes, scope="page", page_index=0)

    assert first == second
    assert image_bytes == original
    for field in (
        "blur",
        "low_contrast",
        "underexposure",
        "overexposure",
        "uneven_illumination",
        "skew",
    ):
        value = getattr(first.profile, field)
        assert value is None or 0.0 <= value <= 1.0


def test_blur_and_low_contrast_move_in_expected_direction() -> None:
    clean = _printed_page()
    blurred = cv2.GaussianBlur(clean, (17, 17), 5)
    low_contrast = np.clip(112 + clean.astype(np.float32) * (35 / 255), 0, 255).astype(np.uint8)

    clean_profile = _profile(clean)
    assert _profile(blurred).blur > clean_profile.blur
    assert _profile(low_contrast).low_contrast > clean_profile.low_contrast


def test_exposure_scores_move_in_expected_direction() -> None:
    clean = _printed_page()
    dark = np.clip(clean.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
    bright = np.clip(204 + clean.astype(np.float32) * 0.20, 0, 255).astype(np.uint8)

    clean_profile = _profile(clean)
    assert _profile(dark).underexposure > clean_profile.underexposure
    assert _profile(bright).overexposure > clean_profile.overexposure


def test_washed_out_page_uses_local_contrast_and_highlight_clipping() -> None:
    clean = _printed_page()
    washed_out = np.clip(170 + clean.astype(np.float32) * 0.33, 0, 255).astype(np.uint8)

    clean_profile = _profile(clean)
    washed_profile = _profile(washed_out)

    assert washed_profile.low_contrast > clean_profile.low_contrast
    assert washed_profile.overexposure > clean_profile.overexposure
    assert washed_profile.raw_measurements["local_content_contrast_median"] < 100
    assert washed_profile.raw_measurements["highlight_clipping_score"] > 0


def test_illumination_score_moves_for_background_gradient() -> None:
    clean = _printed_page()
    gradient = np.linspace(0.35, 1.0, clean.shape[1], dtype=np.float32)
    uneven = np.clip(clean.astype(np.float32) * gradient[None, :], 0, 255).astype(np.uint8)

    assert _profile(uneven).uneven_illumination > _profile(clean).uneven_illumination


def test_calibration_and_metric_version_are_configurable() -> None:
    image_bytes = _encode(_printed_page())
    custom = QualityProfilerSettings(metric_version="quality-test-v9", sharpness_clean=5000.0)
    default_profile = profile_image(image_bytes, scope="page", page_index=0).profile
    custom_profile = profile_image(
        image_bytes,
        scope="page",
        page_index=0,
        config=custom,
    ).profile

    assert custom_profile.metric_version == "quality-test-v9"
    assert custom_profile.blur > default_profile.blur


def test_skew_estimate_moves_for_rotated_page() -> None:
    clean = _printed_page()
    height, width = clean.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 8.0, 1.0)
    rotated = cv2.warpAffine(clean, matrix, (width, height), borderValue=255)

    clean_profile = _profile(clean)
    rotated_profile = _profile(rotated)
    assert rotated_profile.skew is not None
    assert rotated_profile.skew_angle_degrees is not None
    assert abs(rotated_profile.skew_angle_degrees) >= 5.0
    assert clean_profile.skew is None or rotated_profile.skew > clean_profile.skew


def test_blank_tiny_and_corrupt_images_fail_gracefully() -> None:
    blank = profile_image(_encode(np.full((100, 100), 255, dtype=np.uint8)), scope="page", page_index=0)
    tiny = profile_image(_encode(np.full((8, 8), 255, dtype=np.uint8)), scope="page", page_index=0)
    corrupt = profile_image(b"not-an-image", scope="page", page_index=0)

    assert blank.profile.text_density == 0.0
    assert blank.profile.skew is None
    assert any(item.startswith("image_too_small") for item in tiny.profile.warnings)
    assert "image_decode_failed" in corrupt.profile.warnings
    assert all(value is None for value in (tiny.profile.blur, corrupt.profile.blur))


def test_native_pdf_is_input_kind_not_handwriting_or_capture_mode(tmp_path: Path) -> None:
    pdf_path = tmp_path / "native.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Native PDF text")
    document.save(pdf_path)
    document.close()

    pages = profile_pages(render_document_pages(pdf_path), pdf_path)
    profile = pages[0].quality_assessment.profile

    assert profile.input_kind == InputKind.NATIVE_PDF
    assert profile.acquisition_mode == AcquisitionMode.DIGITAL
    assert profile.handwriting_likelihood is None


def test_sparse_ocr_over_full_page_raster_is_scanned_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan-wrapper.pdf"
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    full_page_scan = cv2.resize(_printed_page(), (595, 842))
    page.insert_image(page.rect, stream=_encode(full_page_scan))
    page.insert_text((10, 10), "ocr", render_mode=3)
    document.set_metadata({"author": "CamScanner", "producer": "intsig.com pdf producer"})
    document.save(pdf_path)
    document.close()

    pages = profile_pages(render_document_pages(pdf_path), pdf_path)
    profile = pages[0].quality_assessment.profile

    assert profile.input_kind == InputKind.RASTER_PDF
    assert profile.acquisition_mode == AcquisitionMode.SCANNED
    assert profile.raw_measurements["pdf_max_image_coverage"] >= 0.99
    assert profile.raw_measurements["pdf_text_character_count"] <= 100
    assert "pdf_scan_wrapper:full_page_raster_with_sparse_text" in profile.warnings


def test_page_orientation_is_reported_separately_from_skew(monkeypatch) -> None:
    monkeypatch.setattr(
        "rav_idp.components.quality_profiler.pytesseract.image_to_osd",
        lambda *_args, **_kwargs: {"rotate": 90, "orientation_conf": 6.0},
    )

    profile = _profile(_printed_page())

    assert profile.rotation_required_degrees == 90
    assert profile.orientation_confidence == 0.6
    assert profile.raw_measurements["orientation_confidence_raw"] == 6.0


def test_low_confidence_osd_falls_back_to_ocr_rotation_vote(monkeypatch) -> None:
    monkeypatch.setattr(
        "rav_idp.components.quality_profiler.pytesseract.image_to_osd",
        lambda *_args, **_kwargs: {"rotate": 0, "orientation_conf": 0.2},
    )
    candidates = iter(
        (
            {"text": ["weak"], "conf": ["30"]},
            {"text": ["slightly", "better"], "conf": ["35", "35"]},
            {"text": ["also", "weak"], "conf": ["30", "30"]},
            {"text": ["strong", "orientation", "evidence"], "conf": ["90", "90", "90"]},
        )
    )
    monkeypatch.setattr(
        "rav_idp.components.quality_profiler.pytesseract.image_to_data",
        lambda *_args, **_kwargs: next(candidates),
    )

    profile = _profile(_printed_page())

    assert profile.rotation_required_degrees == 270
    assert profile.orientation_confidence is not None
    assert profile.orientation_confidence > 0.5
    assert "orientation_fallback:ocr_rotation_vote" in profile.warnings


def test_inspection_manifest_contains_raw_and_normalized_quality(tmp_path: Path) -> None:
    image_bytes = _encode(_printed_page())
    page = PageRecord(
        page_index=0,
        quality_class=QualityClass.CLEAN,
        raw_image=image_bytes,
        processed_image=image_bytes,
    )
    region = DetectedRegion(
        region_id="0_0",
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=0, x1=640, y1=420, page=0),
        original_crop=image_bytes,
        quality_class=QualityClass.SCANNED_CLEAN,
        raw_docling_record={"text": "example"},
        page_index=0,
    )
    assessment = profile_region(region)
    region = region.model_copy(update={"quality_assessment": assessment})

    recorder = VisualArtifactRecorder(tmp_path / "artifacts")
    recorder.record_quality([page], [region])
    payload = json.loads((recorder.paths.quality_dir / "regions.json").read_text(encoding="utf-8"))
    profile = payload[0]["quality_assessment"]["profile"]

    assert "raw_measurements" in profile
    assert "sharpness_laplacian_variance" in profile["raw_measurements"]
    assert 0.0 <= profile["blur"] <= 1.0
    assert profile["metric_version"] == "quality-profiler-v2"
    assert payload[0]["quality_assessment"]["entity_type"] == "text"
    assert payload[0]["quality_assessment"]["bbox"]["x1"] == 640
