"""Image comparator."""

from __future__ import annotations

from ...models import DetectedRegion, EntityType, FidelityResult, ImageReconstruction
from ..reconstructors.image import compute_phash


def phash_similarity(hash_a: str, hash_b: str) -> float:
    import imagehash

    first = imagehash.hex_to_hash(hash_a)
    second = imagehash.hex_to_hash(hash_b)
    return 1.0 - (first - second) / 64.0


def compare_image(reconstruction: ImageReconstruction, region: DetectedRegion, threshold: float) -> FidelityResult:
    """Compare image fidelity metrics."""

    original_phash = compute_phash(region.original_crop)
    similarity = max(0.0, min(1.0, phash_similarity(reconstruction.phash_hex, original_phash)))
    sharpness_ratio = min(
        reconstruction.sharpness_crop / reconstruction.sharpness_original,
        1.0,
    ) if reconstruction.sharpness_original else 0.0
    caption_score = 1.0 if reconstruction.caption_found else 0.8
    score = max(0.0, min(1.0, 0.6 * similarity + 0.3 * sharpness_ratio + 0.1 * caption_score))

    return FidelityResult(
        region_id=region.region_id,
        entity_type=EntityType.IMAGE,
        fidelity_score=round(score, 4),
        passed_threshold=score >= threshold,
        threshold_used=threshold,
        component_scores={
            "phash_similarity": round(similarity, 4),
            "sharpness_ratio": round(sharpness_ratio, 4),
            "caption_score": round(caption_score, 4),
        },
        extractor_name="primary",
    )
