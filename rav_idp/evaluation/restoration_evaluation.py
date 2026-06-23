"""Aggregation contracts for restoration and paper-level ablation experiments."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from ..models import AblationVariant, RestorationEvaluationRecord, TransformKind


def default_ablation_variants() -> list[AblationVariant]:
    """Return named variants; execution still requires benchmark data and extractors."""

    photometric = [
        TransformKind.ILLUMINATION_NORMALIZATION,
        TransformKind.CLAHE,
        TransformKind.UNSHARP_MASK,
    ]
    geometric = [TransformKind.ROTATE_ORIENTATION, TransformKind.DESKEW]
    return [
        AblationVariant(name="v1_no_restoration"),
        AblationVariant(name="photometric_only", enabled_transforms=photometric),
        AblationVariant(name="geometric_only", enabled_transforms=geometric),
        AblationVariant(name="deterministic_full", enabled_transforms=[*geometric, *photometric]),
        AblationVariant(
            name="docres_backend",
            enabled_transforms=[*geometric, *photometric],
            backend_name="docres",
        ),
        AblationVariant(
            name="docentr_binarization_backend",
            enabled_transforms=[TransformKind.ADAPTIVE_BINARIZATION],
            backend_name="docentr",
        ),
    ]


def summarize_restoration_records(records: list[RestorationEvaluationRecord]) -> dict[str, dict]:
    """Aggregate only supplied measurements; missing metrics remain explicit."""

    grouped: dict[str, list[RestorationEvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.variant].append(record)

    fields = (
        "delta_cer",
        "delta_layout_f1",
        "delta_table_accuracy",
        "delta_extraction_fidelity",
        "delta_final_trust_rate",
    )
    summary: dict[str, dict] = {}
    for variant, variant_records in grouped.items():
        metrics = {}
        for field in fields:
            values = [getattr(record, field) for record in variant_records]
            measured = [value for value in values if value is not None]
            metrics[field] = mean(measured) if measured else None
        clean_labels = [record.clean_regression for record in variant_records if record.clean_regression is not None]
        summary[variant] = {
            "page_count": len(variant_records),
            "integrity_pass_rate": mean(record.integrity_passed for record in variant_records),
            "clean_regression_rate": mean(clean_labels) if clean_labels else None,
            **metrics,
        }
    return summary
