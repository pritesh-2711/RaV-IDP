"""Controlled synthetic degradation benchmark for profiler calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from ..components.page_renderer import render_document_pages
from ..components.quality_profiler import profile_image
from ..models import AcquisitionMode, InputKind, QualityProfile
from ..utils import image_bytes_to_ndarray, ndarray_to_png_bytes


SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DegradationCase:
    name: str
    severity_field: str
    levels: tuple[float, ...]


CASES = (
    DegradationCase("blur", "blur", (1.0, 2.0, 4.0)),
    DegradationCase("low_contrast", "low_contrast", (0.20, 0.45, 0.70)),
    DegradationCase("underexposure", "underexposure", (0.20, 0.45, 0.70)),
    DegradationCase("overexposure", "overexposure", (20.0, 45.0, 75.0)),
    DegradationCase("uneven_illumination", "uneven_illumination", (0.15, 0.30, 0.50)),
    DegradationCase("skew", "skew", (2.0, 5.0, 8.0)),
)


def _degrade(image: np.ndarray, name: str, level: float) -> np.ndarray:
    if name == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=level, sigmaY=level)
    if name == "low_contrast":
        return np.clip(235.0 + (image.astype(np.float32) - 235.0) * (1.0 - level), 0, 255).astype(np.uint8)
    if name == "underexposure":
        return np.clip(image.astype(np.float32) * (1.0 - level), 0, 255).astype(np.uint8)
    if name == "overexposure":
        return np.clip(image.astype(np.float32) + level, 0, 255).astype(np.uint8)
    if name == "uneven_illumination":
        gradient = np.linspace(1.0 - level, 1.0, image.shape[1], dtype=np.float32)
        return np.clip(image.astype(np.float32) * gradient[None, :, None], 0, 255).astype(np.uint8)
    if name == "skew":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), level, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    raise ValueError(f"Unknown degradation: {name}")


def _profile(image: np.ndarray) -> QualityProfile:
    return profile_image(
        ndarray_to_png_bytes(image),
        scope="page",
        page_index=0,
        input_kind=InputKind.IMAGE,
        acquisition_mode=AcquisitionMode.UNKNOWN,
        enable_orientation=False,
    ).profile


def _document_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_quality_benchmark(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    max_documents: int | None = None,
    max_pages_per_document: int = 3,
) -> dict:
    """Benchmark profiler monotonicity without persisting source filenames."""

    source_dir = Path(input_dir).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if max_documents is not None:
        paths = paths[:max_documents]

    rows: list[dict] = []
    for path in paths:
        document_id = _document_id(path)
        for page in render_document_pages(path)[:max_pages_per_document]:
            image = image_bytes_to_ndarray(page.raw_image)
            if image is None:
                continue
            baseline = _profile(image)
            for case in CASES:
                for level_index, level in enumerate(case.levels, start=1):
                    degraded_profile = _profile(_degrade(image, case.name, level))
                    baseline_score = getattr(baseline, case.severity_field)
                    degraded_score = getattr(degraded_profile, case.severity_field)
                    rows.append(
                        {
                            "document_id": document_id,
                            "page_index": page.page_index,
                            "degradation": case.name,
                            "level_index": level_index,
                            "level_value": level,
                            "metric": case.severity_field,
                            "baseline_score": baseline_score,
                            "degraded_score": degraded_score,
                            "score_delta": (
                                degraded_score - baseline_score
                                if degraded_score is not None and baseline_score is not None
                                else None
                            ),
                            "metric_version": degraded_profile.metric_version,
                        }
                    )

    correlations: dict[str, float | None] = {}
    monotonic_pass_rates: dict[str, float | None] = {}
    for case in CASES:
        case_rows = [row for row in rows if row["degradation"] == case.name]
        groups: dict[tuple[str, int], list[dict]] = {}
        for row in case_rows:
            groups.setdefault((row["document_id"], row["page_index"]), []).append(row)
        group_correlations: list[float] = []
        monotonic_results: list[bool] = []
        for group_rows in groups.values():
            ordered = sorted(group_rows, key=lambda row: row["level_index"])
            scores = [row["degraded_score"] for row in ordered]
            valid_scores = [score for score in scores if score is not None]
            if len(valid_scores) == len(ordered):
                monotonic_results.append(all(right >= left for left, right in zip(scores, scores[1:])))
            if len(valid_scores) >= 3 and len(set(valid_scores)) >= 2:
                correlation = spearmanr(
                    [row["level_index"] for row in ordered],
                    scores,
                ).statistic
                if np.isfinite(correlation):
                    group_correlations.append(float(correlation))
        correlations[case.name] = float(np.mean(group_correlations)) if group_correlations else None
        monotonic_pass_rates[case.name] = (
            float(np.mean(monotonic_results)) if monotonic_results else None
        )

    csv_path = target_dir / "quality_benchmark_rows.csv"
    fieldnames = list(rows[0]) if rows else [
        "document_id", "page_index", "degradation", "level_index", "level_value",
        "metric", "baseline_score", "degraded_score", "score_delta", "metric_version",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "document_count": len(paths),
        "observation_count": len(rows),
        "spearman_by_degradation": correlations,
        "monotonic_pass_rate_by_degradation": monotonic_pass_rates,
        "privacy": "Inputs are identified only by SHA-256; filenames and paths are not persisted.",
    }
    (target_dir / "quality_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
