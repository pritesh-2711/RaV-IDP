"""Text comparator."""

from __future__ import annotations

from Levenshtein import distance as lev_distance

from ...models import EntityType, FidelityResult, TextReconstruction


def compare_text(
    reconstruction: TextReconstruction,
    extracted_text: str,
    region_id: str,
    threshold: float,
    entity_type: EntityType = EntityType.TEXT,
) -> FidelityResult:
    """Compare extracted text to re-OCR or PDF text."""

    reference = reconstruction.reocr_text
    if not reference:
        score = 1.0 if not extracted_text else 0.0
        cer = 0.0 if score == 1.0 else 1.0
    else:
        cer = lev_distance(extracted_text, reference) / len(reference)
        score = max(0.0, 1.0 - cer)

    return FidelityResult(
        region_id=region_id,
        entity_type=entity_type,
        fidelity_score=round(score, 4),
        passed_threshold=score >= threshold,
        threshold_used=threshold,
        component_scores={"cer": round(cer, 4)},
        extractor_name="primary",
    )
