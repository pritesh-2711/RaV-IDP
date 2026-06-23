"""Experimental orchestration for versioned page evidence."""

from __future__ import annotations

from ..config import RestorationSettings, get_settings
from ..models import PageEvidence, PageRecord
from ..utils import image_bytes_to_ndarray
from .page_restorer import identity_mapping
from .restoration_backends import DeterministicRestorationBackend, RestorationBackend
from .restoration_integrity import evaluate_restoration_integrity
from .restoration_planner import plan_page_restoration


def build_page_evidence(
    pages: list[PageRecord],
    *,
    enable_restoration: bool = False,
    config: RestorationSettings | None = None,
    backend: RestorationBackend | None = None,
) -> list[PageEvidence]:
    """Build v1/v2 evidence; disabled mode guarantees byte-identical v2 pages."""

    settings = config or get_settings().restoration
    selected_backend = backend or DeterministicRestorationBackend()
    bundles: list[PageEvidence] = []
    for page in pages:
        candidate_page_v2: bytes | None = None
        plan = plan_page_restoration(page, settings)
        if not enable_restoration:
            plan = plan.model_copy(
                update={
                    "operations": [],
                    "warnings": [*plan.warnings, "restoration_disabled:identity_v2"],
                }
            )
            image = image_bytes_to_ndarray(page.raw_image)
            if image is None:
                raise ValueError("Unable to decode source page evidence.")
            height, width = image.shape[:2]
            working_page_v2 = page.raw_image
            mapping = identity_mapping(width, height)
        else:
            if plan.operations:
                backend_result = selected_backend.restore(page.raw_image, plan)
                working_page_v2 = backend_result.image_bytes
                candidate_page_v2 = backend_result.image_bytes
                mapping = backend_result.mapping
                if backend_result.warnings:
                    plan = plan.model_copy(
                        update={"warnings": [*plan.warnings, *backend_result.warnings]}
                    )
            else:
                image = image_bytes_to_ndarray(page.raw_image)
                if image is None:
                    raise ValueError("Unable to decode source page evidence.")
                height, width = image.shape[:2]
                working_page_v2 = page.raw_image
                mapping = identity_mapping(width, height)
                backend_result = None
        if mapping is None:
            image = image_bytes_to_ndarray(page.raw_image)
            height, width = image.shape[:2]
            integrity = evaluate_restoration_integrity(
                page.raw_image,
                page.raw_image,
                identity_mapping(width, height),
                page_index=page.page_index,
                threshold=settings.integrity_threshold,
            ).model_copy(
                update={
                    "passed": False,
                    "warnings": ["restoration_mapping_unavailable"],
                }
            )
            working_page_v2 = page.raw_image
            mapping = identity_mapping(width, height)
            plan = plan.model_copy(
                update={"warnings": [*plan.warnings, "restoration_rejected:mapping_unavailable"]}
            )
        else:
            integrity = evaluate_restoration_integrity(
                page.raw_image,
                working_page_v2,
                mapping,
                page_index=page.page_index,
                threshold=settings.integrity_threshold,
            )
        if not integrity.passed:
            image = image_bytes_to_ndarray(page.raw_image)
            height, width = image.shape[:2]
            working_page_v2 = page.raw_image
            mapping = identity_mapping(width, height)
            plan = plan.model_copy(
                update={"warnings": [*plan.warnings, "restoration_rejected:integrity_failed"]}
            )
        bundles.append(
            PageEvidence(
                page_index=page.page_index,
                source_page_v1=page.raw_image,
                candidate_page_v2=candidate_page_v2,
                working_page_v2=working_page_v2,
                plan=plan,
                mapping_v1_to_v2=mapping,
                integrity=integrity,
                backend_name=(
                    backend_result.backend_name
                    if enable_restoration and backend_result is not None
                    else selected_backend.name
                ),
                backend_task=(
                    backend_result.task
                    if enable_restoration and backend_result is not None
                    else None
                ),
            )
        )
    return bundles
