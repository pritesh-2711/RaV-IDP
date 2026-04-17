"""Visual artifact recording for end-to-end pipeline runs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .io import ensure_parent
from .models import (
    DetectedRegion,
    EntityRecord,
    EntityType,
    PageRecord,
    PipelineTraceRecord,
)
from .utils import image_bytes_to_pil


_ENTITY_COLORS = {
    EntityType.TEXT: "#1f77b4",
    EntityType.TABLE: "#2ca02c",
    EntityType.IMAGE: "#d62728",
    EntityType.FORMULA: "#9467bd",
    EntityType.URL: "#ff7f0e",
}


def _safe_json(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    return str(value)


@dataclass
class RunArtifactPaths:
    run_dir: Path
    pages_dir: Path
    layout_dir: Path
    quality_dir: Path
    preprocessed_dir: Path
    traces_dir: Path
    final_dir: Path


class BaseArtifactRecorder:
    """Base class for writing structured pipeline artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        root = Path(run_dir).expanduser().resolve()
        self.paths = RunArtifactPaths(
            run_dir=root,
            pages_dir=root / "00_pages",
            layout_dir=root / "01_layout",
            quality_dir=root / "02_quality_classification",
            preprocessed_dir=root / "03_preprocessed_regions",
            traces_dir=root / "04_rav_traces",
            final_dir=root / "05_final_output",
        )
        for path in self.paths.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self.paths.run_dir

    def write_json(self, path: str | Path, payload: object) -> Path:
        target = ensure_parent(path)
        target.write_text(json.dumps(_safe_json(payload), indent=2), encoding="utf-8")
        return target

    def write_text(self, path: str | Path, text: str) -> Path:
        target = ensure_parent(path)
        target.write_text(text, encoding="utf-8")
        return target

    def write_image_bytes(self, path: str | Path, image_bytes: bytes) -> Path | None:
        if not image_bytes:
            return None
        target = ensure_parent(path)
        target.write_bytes(image_bytes)
        return target


class VisualArtifactRecorder(BaseArtifactRecorder):
    """Recorder that stores visual and structured outputs for each pipeline step."""

    def write_run_manifest(self, document_path: str | Path) -> None:
        self.write_json(
            self.run_dir / "run_manifest.json",
            {
                "document_path": str(Path(document_path).expanduser().resolve()),
                "run_dir": str(self.run_dir),
                "stages": {
                    "00_pages": "Rendered page images before layout analysis.",
                    "01_layout": "Layout detection overlays and region metadata.",
                    "02_quality_classification": "Region-level quality labels and original crops.",
                    "03_preprocessed_regions": "Processed region crops after preprocessing.",
                    "04_rav_traces": "Primary extraction, fallback, reconstruction, and fidelity per region.",
                    "05_final_output": "Final overlays, entity records, and run summary.",
                },
            },
        )

    def record_pages(self, page_records: list[PageRecord]) -> None:
        manifest = []
        for page in page_records:
            page_stem = f"page_{page.page_index:03d}"
            self.write_image_bytes(self.paths.pages_dir / f"{page_stem}_raw.png", page.raw_image)
            self.write_image_bytes(self.paths.pages_dir / f"{page_stem}_processed.png", page.processed_image)
            manifest.append(page.model_dump(mode="json", exclude={"raw_image", "processed_image"}))
        self.write_json(self.paths.pages_dir / "pages.json", manifest)

    def _draw_overlay(
        self,
        page_records: list[PageRecord],
        regions: list[DetectedRegion],
        output_dir: Path,
        labeler,
    ) -> None:
        regions_by_page: dict[int, list[DetectedRegion]] = {}
        for region in regions:
            regions_by_page.setdefault(region.page_index, []).append(region)

        for page in page_records:
            image = image_bytes_to_pil(page.raw_image).convert("RGBA")
            draw = ImageDraw.Draw(image)
            for region in regions_by_page.get(page.page_index, []):
                color = _ENTITY_COLORS.get(region.entity_type, "#333333")
                draw.rectangle(
                    [region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1],
                    outline=color,
                    width=3,
                )
                label = labeler(region)
                if label:
                    x0 = max(0, int(region.bbox.x0))
                    y0 = max(0, int(region.bbox.y0) - 18)
                    text_width = max(60, 7 * len(label))
                    draw.rectangle([x0, y0, x0 + text_width, y0 + 18], fill=color)
                    draw.text((x0 + 4, y0 + 2), label, fill="white")
            image.convert("RGB").save(output_dir / f"page_{page.page_index:03d}_overlay.png", format="PNG")

    def record_layout(self, page_records: list[PageRecord], regions: list[DetectedRegion]) -> None:
        self._draw_overlay(
            page_records,
            regions,
            self.paths.layout_dir,
            lambda region: f"{region.region_id}:{region.entity_type.value}",
        )
        payload = [
            region.model_dump(mode="json", exclude={"original_crop", "processed_crop"})
            for region in regions
        ]
        self.write_json(self.paths.layout_dir / "regions.json", payload)

    def record_quality(self, page_records: list[PageRecord], regions: list[DetectedRegion]) -> None:
        self._draw_overlay(
            page_records,
            regions,
            self.paths.quality_dir,
            lambda region: f"{region.region_id}:{region.entity_type.value}|{(region.quality_class.value if region.quality_class else 'unknown')}",
        )
        manifest = []
        for region in regions:
            region_dir = self.paths.quality_dir / f"page_{region.page_index:03d}" / region.region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            self.write_image_bytes(region_dir / "original_crop.png", region.original_crop)
            manifest.append(region.model_dump(mode="json", exclude={"original_crop", "processed_crop"}))
        self.write_json(self.paths.quality_dir / "regions.json", manifest)

    def record_preprocessed(self, regions: list[DetectedRegion]) -> None:
        manifest = []
        for region in regions:
            region_dir = self.paths.preprocessed_dir / f"page_{region.page_index:03d}" / region.region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            self.write_image_bytes(region_dir / "original_crop.png", region.original_crop)
            self.write_image_bytes(region_dir / "processed_crop.png", region.processed_crop or region.original_crop)
            manifest.append(region.model_dump(mode="json", exclude={"original_crop", "processed_crop"}))
        self.write_json(self.paths.preprocessed_dir / "regions.json", manifest)

    def _entity_summary(self, trace_entity) -> dict:
        payload = trace_entity.model_dump(mode="python")
        content = payload.get("content", {})
        if "crop_bytes" in content:
            content.pop("crop_bytes", None)
        return payload

    def _reconstruction_summary(self, reconstruction) -> dict:
        payload = reconstruction.model_dump(mode="python")
        content = payload.get("content", {})
        content.pop("rendered_image", None)
        return payload

    def record_trace(self, region: DetectedRegion, trace: PipelineTraceRecord) -> None:
        region_dir = self.paths.traces_dir / f"page_{region.page_index:03d}" / region.region_id
        region_dir.mkdir(parents=True, exist_ok=True)

        self.write_image_bytes(region_dir / "original_crop.png", region.original_crop)
        self.write_image_bytes(region_dir / "processed_crop.png", region.processed_crop or region.original_crop)

        if trace.primary_entity.entity_type == EntityType.IMAGE:
            self.write_image_bytes(region_dir / "primary_extracted_image.png", trace.primary_entity.content.crop_bytes)
        if trace.fallback_entity and trace.fallback_entity.entity_type == EntityType.IMAGE:
            self.write_image_bytes(region_dir / "fallback_extracted_image.png", trace.fallback_entity.content.crop_bytes)
        if trace.primary_reconstruction.entity_type == EntityType.TABLE:
            self.write_image_bytes(region_dir / "primary_reconstructed_table.png", trace.primary_reconstruction.content.rendered_image)
        if trace.fallback_reconstruction and trace.fallback_reconstruction.entity_type == EntityType.TABLE:
            self.write_image_bytes(region_dir / "fallback_reconstructed_table.png", trace.fallback_reconstruction.content.rendered_image)

        trace_payload = {
            "region": region.model_dump(mode="json", exclude={"original_crop", "processed_crop"}),
            "primary_entity": self._entity_summary(trace.primary_entity),
            "primary_reconstruction": self._reconstruction_summary(trace.primary_reconstruction),
            "primary_fidelity": trace.primary_fidelity.model_dump(mode="json"),
            "fallback_entity": self._entity_summary(trace.fallback_entity) if trace.fallback_entity else None,
            "fallback_reconstruction": self._reconstruction_summary(trace.fallback_reconstruction) if trace.fallback_reconstruction else None,
            "fallback_fidelity": trace.fallback_fidelity.model_dump(mode="json") if trace.fallback_fidelity else None,
            "final_entity": self._entity_summary(trace.final_entity),
            "final_fidelity": trace.final_fidelity.model_dump(mode="json"),
            "provenance": trace.provenance.model_dump(mode="json"),
            "context_text": trace.context_text,
        }
        self.write_json(region_dir / "trace.json", trace_payload)

        summary_lines = [
            f"Region: {region.region_id}",
            f"Entity type: {region.entity_type.value}",
            f"Quality class: {region.quality_class.value if region.quality_class else 'unknown'}",
            f"Primary extractor: {trace.primary_entity.extractor_name}",
            f"Primary fidelity: {trace.primary_fidelity.fidelity_score:.4f}",
            f"Fallback triggered: {'yes' if trace.provenance.fallback_triggered else 'no'}",
            f"Final extractor: {trace.provenance.final_extractor}",
            f"Final fidelity: {trace.final_fidelity.fidelity_score:.4f}",
            f"Low confidence: {'yes' if trace.provenance.low_confidence_flag else 'no'}",
        ]
        self.write_text(region_dir / "summary.txt", "\n".join(summary_lines) + "\n")

    def record_final_output(self, page_records: list[PageRecord], entity_records: list[EntityRecord]) -> None:
        record_map = {record.region_id: record for record in entity_records}

        def labeler(region: DetectedRegion) -> str:
            record = record_map.get(region.region_id)
            if record is None:
                return region.region_id
            return f"{region.region_id}:{record.entity_type.value}|{record.fidelity_score:.2f}"

        regions = [
            DetectedRegion(
                region_id=record.region_id,
                entity_type=record.entity_type,
                bbox=record.bbox,
                original_crop=b"",
                processed_crop=b"",
                raw_docling_record={},
                page_index=record.page_index,
            )
            for record in entity_records
        ]
        self._draw_overlay(page_records, regions, self.paths.final_dir, labeler)

        serialized_records = []
        for record in entity_records:
            payload = record.model_dump(mode="python")
            content = payload.get("content", {})
            if isinstance(content, dict):
                content.pop("crop_bytes", None)
            serialized_records.append(payload)
        self.write_json(
            self.paths.final_dir / "entity_records.json",
            serialized_records,
        )

        counts = Counter(record.entity_type.value for record in entity_records)
        low_confidence = sum(1 for record in entity_records if record.low_confidence_flag)
        self.write_json(
            self.paths.final_dir / "run_summary.json",
            {
                "num_entities": len(entity_records),
                "entity_counts": dict(counts),
                "low_confidence_count": low_confidence,
            },
        )
