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
    )


def as_path(value: str | Path) -> Path:
    """Normalize a string or path into a Path instance."""

    return value if isinstance(value, Path) else Path(value)
