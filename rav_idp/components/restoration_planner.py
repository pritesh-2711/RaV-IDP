"""Deterministic experimental planner for page-level restoration."""

from __future__ import annotations

from ..config import RestorationSettings, get_settings
from ..models import (
    AcquisitionMode,
    PageRecord,
    RestorationPlan,
    TransformKind,
    TransformSpec,
)


def plan_page_restoration(
    page: PageRecord,
    config: RestorationSettings | None = None,
) -> RestorationPlan:
    """Convert a page quality profile into an ordered, auditable plan."""

    settings = config or get_settings().restoration
    assessment = page.quality_assessment
    if assessment is None:
        return RestorationPlan(
            page_index=page.page_index,
            planner_version=settings.planner_version,
            warnings=["restoration_skipped:quality_assessment_missing"],
        )

    profile = assessment.profile
    operations: list[TransformSpec] = []
    warnings: list[str] = []

    if (
        profile.rotation_required_degrees not in (None, 0)
        and (profile.orientation_confidence or 0.0) >= settings.minimum_orientation_confidence
    ):
        operations.append(
            TransformSpec(
                kind=TransformKind.ROTATE_ORIENTATION,
                parameters={"degrees_clockwise": profile.rotation_required_degrees},
                reason="gross page orientation detected with sufficient confidence",
            )
        )
    elif profile.rotation_required_degrees not in (None, 0):
        warnings.append("rotation_skipped:orientation_confidence_below_threshold")

    if (
        profile.skew is not None
        and profile.skew >= settings.skew_trigger
        and profile.skew_angle_degrees is not None
    ):
        operations.append(
            TransformSpec(
                kind=TransformKind.DESKEW,
                parameters={"angle_degrees": profile.skew_angle_degrees},
                reason="minor skew severity exceeded configured threshold",
            )
        )

    if profile.uneven_illumination is not None and profile.uneven_illumination >= settings.illumination_trigger:
        operations.append(
            TransformSpec(
                kind=TransformKind.ILLUMINATION_NORMALIZATION,
                parameters={},
                reason="background illumination variation exceeded configured threshold",
            )
        )

    if profile.low_contrast is not None and profile.low_contrast >= settings.contrast_trigger:
        operations.append(
            TransformSpec(
                kind=TransformKind.CLAHE,
                parameters={"clip_limit": settings.clahe_clip_limit},
                reason="local content contrast fell below configured range",
            )
        )

    if profile.blur is not None and profile.blur >= settings.blur_trigger:
        operations.append(
            TransformSpec(
                kind=TransformKind.UNSHARP_MASK,
                parameters={"amount": settings.unsharp_amount},
                reason="blur severity exceeded configured threshold",
            )
        )

    if profile.acquisition_mode == AcquisitionMode.DIGITAL and operations:
        warnings.append("digital_page_restoration_requires_clean_regression_check")

    return RestorationPlan(
        page_index=page.page_index,
        operations=operations,
        planner_version=settings.planner_version,
        warnings=warnings,
    )


def plan_page_restorations(
    pages: list[PageRecord],
    config: RestorationSettings | None = None,
) -> list[RestorationPlan]:
    return [plan_page_restoration(page, config=config) for page in pages]
