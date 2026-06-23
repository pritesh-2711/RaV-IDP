"""Configuration helpers for RaV-IDP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

DEFAULT_DPI = 150


@dataclass(frozen=True)
class QualityProfilerSettings:
    """Calibration hypotheses for the deterministic Phase 1 profiler."""

    metric_version: str = "quality-profiler-v2"
    min_dimension_px: int = 24
    sharpness_blurred: float = 30.0
    sharpness_clean: float = 500.0
    contrast_low: float = 15.0
    contrast_clean: float = 180.0
    underexposed_background: float = 100.0
    normal_background: float = 220.0
    normal_foreground: float = 140.0
    overexposed_foreground: float = 230.0
    illumination_std_severe: float = 45.0
    skew_severe_degrees: float = 12.0
    minimum_skew_confidence: float = 0.15
    orientation_confidence_reference: float = 10.0
    full_page_raster_coverage: float = 0.80
    sparse_pdf_text_chars: int = 100


@dataclass(frozen=True)
class RestorationSettings:
    """Experimental restoration routing; disabled in the production pipeline."""

    planner_version: str = "restoration-planner-v1"
    blur_trigger: float = 0.55
    contrast_trigger: float = 0.45
    illumination_trigger: float = 0.45
    skew_trigger: float = 0.15
    minimum_orientation_confidence: float = 0.50
    integrity_threshold: float = 0.60
    clahe_clip_limit: float = 2.0
    unsharp_amount: float = 0.35


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: str | None
    openai_model: str          # vision/extraction tasks (fallback extractor, image enricher)
    openai_qa_model: str       # Stage 6 QA — text-only, cheaper model is sufficient
    openai_vision_max_tokens: int
    threshold_table: float
    threshold_image: float
    threshold_text: float
    crop_scale: int
    caption_proximity_px: int
    data_root: Path
    results_root: Path
    quality_profiler: QualityProfilerSettings
    restoration: RestorationSettings
    render_dpi: int = DEFAULT_DPI

    @property
    def threshold_by_type(self) -> dict[str, float]:
        return {
            "table": self.threshold_table,
            "image": self.threshold_image,
            "text": self.threshold_text,
            "formula": self.threshold_text,
            "url": self.threshold_text,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        openai_qa_model=os.getenv("OPENAI_QA_MODEL", "gpt-4.1-mini"),
        openai_vision_max_tokens=int(os.getenv("OPENAI_VISION_MAX_TOKENS", "1024")),
        threshold_table=float(os.getenv("RAV_THRESHOLD_TABLE", "0.75")),
        threshold_image=float(os.getenv("RAV_THRESHOLD_IMAGE", "0.70")),
        threshold_text=float(os.getenv("RAV_THRESHOLD_TEXT", "0.85")),
        crop_scale=int(os.getenv("RAV_CROP_SCALE", "2")),
        caption_proximity_px=int(os.getenv("RAV_CAPTION_PROXIMITY_PX", "60")),
        data_root=Path(os.getenv("RAV_DATA_ROOT", "data")).expanduser().resolve(),
        results_root=Path(os.getenv("RAV_RESULTS_ROOT", "artifacts")).expanduser().resolve(),
        quality_profiler=QualityProfilerSettings(
            metric_version=os.getenv("RAV_QUALITY_METRIC_VERSION", "quality-profiler-v2"),
            min_dimension_px=int(os.getenv("RAV_QUALITY_MIN_DIMENSION_PX", "24")),
            sharpness_blurred=float(os.getenv("RAV_QUALITY_SHARPNESS_BLURRED", "30")),
            sharpness_clean=float(os.getenv("RAV_QUALITY_SHARPNESS_CLEAN", "500")),
            contrast_low=float(os.getenv("RAV_QUALITY_CONTRAST_LOW", "15")),
            contrast_clean=float(os.getenv("RAV_QUALITY_CONTRAST_CLEAN", "180")),
            underexposed_background=float(os.getenv("RAV_QUALITY_UNDEREXPOSED_BACKGROUND", "100")),
            normal_background=float(os.getenv("RAV_QUALITY_NORMAL_BACKGROUND", "220")),
            normal_foreground=float(os.getenv("RAV_QUALITY_NORMAL_FOREGROUND", "140")),
            overexposed_foreground=float(os.getenv("RAV_QUALITY_OVEREXPOSED_FOREGROUND", "230")),
            illumination_std_severe=float(os.getenv("RAV_QUALITY_ILLUMINATION_STD_SEVERE", "45")),
            skew_severe_degrees=float(os.getenv("RAV_QUALITY_SKEW_SEVERE_DEGREES", "12")),
            minimum_skew_confidence=float(os.getenv("RAV_QUALITY_MIN_SKEW_CONFIDENCE", "0.15")),
            orientation_confidence_reference=float(
                os.getenv("RAV_QUALITY_ORIENTATION_CONFIDENCE_REFERENCE", "10")
            ),
            full_page_raster_coverage=float(
                os.getenv("RAV_QUALITY_FULL_PAGE_RASTER_COVERAGE", "0.80")
            ),
            sparse_pdf_text_chars=int(os.getenv("RAV_QUALITY_SPARSE_PDF_TEXT_CHARS", "100")),
        ),
        restoration=RestorationSettings(
            planner_version=os.getenv("RAV_RESTORATION_PLANNER_VERSION", "restoration-planner-v1"),
            blur_trigger=float(os.getenv("RAV_RESTORATION_BLUR_TRIGGER", "0.55")),
            contrast_trigger=float(os.getenv("RAV_RESTORATION_CONTRAST_TRIGGER", "0.45")),
            illumination_trigger=float(os.getenv("RAV_RESTORATION_ILLUMINATION_TRIGGER", "0.45")),
            skew_trigger=float(os.getenv("RAV_RESTORATION_SKEW_TRIGGER", "0.15")),
            minimum_orientation_confidence=float(
                os.getenv("RAV_RESTORATION_MIN_ORIENTATION_CONFIDENCE", "0.50")
            ),
            integrity_threshold=float(os.getenv("RAV_RESTORATION_INTEGRITY_THRESHOLD", "0.60")),
            clahe_clip_limit=float(os.getenv("RAV_RESTORATION_CLAHE_CLIP_LIMIT", "2.0")),
            unsharp_amount=float(os.getenv("RAV_RESTORATION_UNSHARP_AMOUNT", "0.35")),
        ),
    )


def as_path(value: str | Path) -> Path:
    """Normalize a string or path into a Path instance."""

    return value if isinstance(value, Path) else Path(value)
