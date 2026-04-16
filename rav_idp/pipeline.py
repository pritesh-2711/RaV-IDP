"""Top-level RaV-IDP pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .components.comparators.image import compare_image
from .components.comparators.table import compare_table
from .components.comparators.text import compare_text
from .components.context_enricher import enrich_context
from .components.entity_router import route_entities
from .components.extractors.image import extract_image
from .components.extractors.table import extract_table
from .components.extractors.text import extract_text
from .components.fallback_extractor import call_vision_fallback
from .components.layout_detector import detect_layout
from .components.page_renderer import render_document_pages
from .components.region_preprocessor import preprocess_regions
from .components.region_quality_classifier import classify_regions
from .components.reconstructors.image import reconstruct_image
from .components.reconstructors.table import reconstruct_table
from .components.reconstructors.text import reconstruct_text
from .config import get_settings
from .models import (
    ContextRecord,
    DetectedRegion,
    EntityRecord,
    EntityType,
    ExtractedEntity,
    FidelityResult,
    ProvenanceRecord,
)
from .utils import is_native_pdf


ExtractorFn = Callable[..., ExtractedEntity]
ReconstructorFn = Callable[..., object]
ComparatorFn = Callable[..., FidelityResult]


@dataclass
class ProcessorBundle:
    extractor: ExtractorFn
    reconstructor: ReconstructorFn
    comparator: ComparatorFn
    threshold: float


class RaVIDPPipeline:
    """End-to-end RaV-IDP pipeline."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.bundles: dict[EntityType, ProcessorBundle] = {
            EntityType.TABLE: ProcessorBundle(
                extractor=extract_table,
                reconstructor=reconstruct_table,
                comparator=compare_table,
                threshold=settings.threshold_table,
            ),
            EntityType.IMAGE: ProcessorBundle(
                extractor=extract_image,
                reconstructor=reconstruct_image,
                comparator=compare_image,
                threshold=settings.threshold_image,
            ),
            EntityType.TEXT: ProcessorBundle(
                extractor=extract_text,
                reconstructor=reconstruct_text,
                comparator=compare_text,
                threshold=settings.threshold_text,
            ),
            EntityType.FORMULA: ProcessorBundle(
                extractor=extract_text,
                reconstructor=reconstruct_text,
                comparator=compare_text,
                threshold=settings.threshold_text,
            ),
            EntityType.URL: ProcessorBundle(
                extractor=extract_text,
                reconstructor=reconstruct_text,
                comparator=compare_text,
                threshold=settings.threshold_text,
            ),
        }

    def _context_text(self, region: DetectedRegion, page_regions: list[DetectedRegion]) -> str:
        context = enrich_context(
            entity=ExtractedEntity(
                region_id=region.region_id,
                entity_type=region.entity_type,
                content=extract_text(region).content if region.entity_type != EntityType.IMAGE else extract_text(region.model_copy(update={"entity_type": EntityType.TEXT, "raw_docling_record": {"text": ""}})).content,
                extractor_name="primary",
            ),
            region=region,
            all_regions=page_regions,
        )
        lines = [context.caption_text or ""]
        lines.extend(context.preceding_text)
        lines.extend(context.following_text)
        return "\n".join(line for line in lines if line.strip())

    def rav_loop(
        self,
        region: DetectedRegion,
        page_regions: list[DetectedRegion],
        document_path: str | Path,
    ) -> tuple[ExtractedEntity, FidelityResult, ProvenanceRecord]:
        """Run primary extraction and fallback when needed."""

        bundle = self.bundles[region.entity_type]
        extra_kwargs = self._bundle_kwargs(region, page_regions, document_path)
        entity = bundle.extractor(region, **extra_kwargs["extractor"])
        reconstruction = bundle.reconstructor(entity, region, **extra_kwargs["reconstructor"])
        fidelity = self._compare(bundle, region, entity, reconstruction, "primary")
        provenance = ProvenanceRecord(
            region_id=region.region_id,
            primary_fidelity=fidelity.fidelity_score,
            final_extractor=entity.extractor_name,
            final_fidelity=fidelity.fidelity_score,
            low_confidence_flag=not fidelity.passed_threshold,
        )

        if fidelity.passed_threshold:
            return entity, fidelity, provenance

        fallback_context = self._context_text(region, page_regions)
        fallback_entity = call_vision_fallback(region, context_text=fallback_context)
        fallback_reconstruction = bundle.reconstructor(
            fallback_entity,
            region,
            **extra_kwargs["reconstructor"],
        )
        fallback_fidelity = self._compare(bundle, region, fallback_entity, fallback_reconstruction, "fallback")

        best_entity = entity
        best_fidelity = fidelity
        if fallback_fidelity.fidelity_score >= fidelity.fidelity_score:
            best_entity = fallback_entity
            best_fidelity = fallback_fidelity

        provenance = ProvenanceRecord(
            region_id=region.region_id,
            primary_fidelity=fidelity.fidelity_score,
            fallback_triggered=True,
            fallback_fidelity=fallback_fidelity.fidelity_score,
            final_extractor=best_entity.extractor_name,
            final_fidelity=best_fidelity.fidelity_score,
            low_confidence_flag=not best_fidelity.passed_threshold,
        )
        return best_entity, best_fidelity, provenance

    def _bundle_kwargs(
        self,
        region: DetectedRegion,
        page_regions: list[DetectedRegion],
        document_path: str | Path,
    ) -> dict[str, dict]:
        native_pdf = is_native_pdf(document_path)
        if region.entity_type == EntityType.IMAGE:
            return {
                "extractor": {"document_path": document_path, "scale": self.settings.crop_scale},
                "reconstructor": {"all_regions": page_regions, "caption_proximity_px": self.settings.caption_proximity_px},
            }
        if region.entity_type in {EntityType.TEXT, EntityType.FORMULA, EntityType.URL}:
            return {
                "extractor": {},
                "reconstructor": {"is_native_pdf": native_pdf, "document_path": document_path},
            }
        return {"extractor": {}, "reconstructor": {}}

    def _compare(
        self,
        bundle: ProcessorBundle,
        region: DetectedRegion,
        entity: ExtractedEntity,
        reconstruction: object,
        extractor_name: str,
    ) -> FidelityResult:
        if region.entity_type in {EntityType.TEXT, EntityType.FORMULA, EntityType.URL}:
            result = bundle.comparator(
                reconstruction.content,
                entity.content.text,
                region.region_id,
                bundle.threshold,
                entity_type=region.entity_type,
            )
        else:
            result = bundle.comparator(reconstruction.content, region, bundle.threshold)
        result.extractor_name = extractor_name
        return result

    def run(self, document_path: str | Path) -> list[EntityRecord]:
        """Run the full pipeline and return final entity records."""

        page_records = render_document_pages(document_path)
        regions = detect_layout(document_path, page_records)
        regions = preprocess_regions(classify_regions(regions))
        route_entities(regions)

        entity_records: list[EntityRecord] = []
        for region in regions:
            page_regions = [candidate for candidate in regions if candidate.page_index == region.page_index]
            entity, fidelity, provenance = self.rav_loop(region, page_regions, document_path)
            context = enrich_context(entity, region, page_regions)
            entity_records.append(
                EntityRecord(
                    region_id=region.region_id,
                    page_index=region.page_index,
                    entity_type=region.entity_type,
                    bbox=region.bbox,
                    content=entity.content,
                    fidelity_score=fidelity.fidelity_score,
                    low_confidence_flag=provenance.low_confidence_flag,
                    context=context,
                    provenance=provenance,
                )
            )
        return entity_records
