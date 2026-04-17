"""Stage 3b: image extraction benchmark on ScanBank.

Dataset: WKLI22/scanbank_hf (HuggingFace, MIT)
Each row is a document page image with COCO-style bounding box annotations
for embedded figures. Columns: image_id, image, width, height,
objects {area, bbox [x,y,w,h], category, id}.

Evaluation design:
  For each annotated figure bbox, crop the region from the page image, run
  reconstruct_image + compare_image, and record fidelity metrics. extract_image
  is bypassed here because it expects a PDF path; for raster input the
  extraction is the crop itself.

  Using GT bboxes isolates the extractor component from layout detection
  (tested in stage 2). pHash similarity between the crop and its own source
  region is expected to be near 1.0 — confirming the extraction is lossless.
  The informative metrics are sharpness ratio and pass rate distribution.

Metrics reported per sample and in summary:
  - extraction_success_rate : non-degenerate crops / total annotated bboxes
  - mean_fidelity           : average fidelity score across all extracted regions
  - pass_rate               : fraction of regions with fidelity >= threshold
  - mean_sharpness_crop     : Laplacian variance of extracted crops
  - mean_sharpness_ratio    : sharpness_crop / sharpness_original (capped at 1.0)
  - caption_detection_rate  : fraction of regions where adjacent caption was found
                              (always 0.0 in standalone eval — no surrounding regions)
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from ..components.comparators.image import compare_image
from ..components.image_enricher import enrich_image
from ..components.reconstructors.image import reconstruct_image
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType, ExtractedEntity, ImageContent


def _crop_bytes(page_image_bytes: bytes, bbox_xywh: list[float], page_w: int, page_h: int) -> bytes:
    """Crop a COCO-style [x, y, w, h] bbox from a page image. Returns PNG bytes."""
    x, y, w, h = bbox_xywh
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(page_w, int(x + w)), min(page_h, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return b""
    image = Image.open(io.BytesIO(page_image_bytes)).convert("RGB")
    crop = image.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_bbox_xywh(bbox_xywh: list[float]) -> list[float]:
    return [float(value) for value in bbox_xywh]


def _make_region(region_id: str, crop: bytes, bbox_xywh: list[float]) -> DetectedRegion:
    x, y, w, h = _normalize_bbox_xywh(bbox_xywh)
    return DetectedRegion(
        region_id=region_id,
        entity_type=EntityType.IMAGE,
        bbox=BoundingBox(x0=x, y0=y, x1=x + w, y1=y + h, page=0),
        original_crop=crop,
        processed_crop=crop,
        raw_docling_record={},
        page_index=0,
    )


@dataclass
class ImageBenchmarkRecord:
    sample_id: str
    region_id: str
    bbox: list[float]
    extraction_success: bool
    fidelity_score: float
    passed_threshold: bool
    phash_similarity: float
    sharpness_crop: float
    sharpness_original: float
    sharpness_ratio: float
    caption_found: bool
    enrichment_attempted: bool
    enrichment_populated: bool
    image_type: str | None
    has_description: bool
    has_extracted_text: bool
    has_structured_data: bool


@dataclass
class ImageBenchmarkSummary:
    split: str
    num_pages: int
    num_regions: int
    extraction_success_rate: float
    mean_fidelity: float
    pass_rate: float
    mean_sharpness_crop: float
    mean_sharpness_ratio: float
    caption_detection_rate: float
    enrichment_attempt_rate: float
    enrichment_success_rate: float
    description_coverage: float
    extracted_text_coverage: float
    structured_data_coverage: float


def run_image_benchmark(
    dataset_root: str | Path,
    split: str = "test",
    limit: int | None = None,
    enrich_limit: int = 20,
) -> tuple[ImageBenchmarkSummary, list[ImageBenchmarkRecord]]:
    """Run image extraction benchmark on ScanBank parquet data."""

    settings = get_settings()
    dataset_root = Path(dataset_root)

    # ScanBank parquet files are stored under data/ in HuggingFace layout
    parquet_candidates = list(dataset_root.glob(f"data/{split}*.parquet"))
    if not parquet_candidates:
        parquet_candidates = list(dataset_root.glob(f"{split}*.parquet"))
    if not parquet_candidates:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' under {dataset_root}. "
            "Run: python -m rav_idp.data fetch scanbank"
        )

    frame = pd.read_parquet(parquet_candidates[0])
    if limit is not None:
        frame = frame.head(limit)

    records: list[ImageBenchmarkRecord] = []
    enrich_count = 0

    for row in frame.itertuples(index=False):
        # image column is either a dict with "bytes" key (HuggingFace format) or raw bytes
        image_col = row.image
        page_image_bytes = image_col["bytes"] if isinstance(image_col, dict) else image_col
        page_w = int(row.width)
        page_h = int(row.height)
        sample_id = str(row.image_id)

        objects = row.objects if hasattr(row, "objects") else {}
        bboxes = objects.get("bbox", []) if objects else []
        obj_ids: list[int] = objects.get("id", list(range(len(bboxes)))) if objects else []

        if len(bboxes) == 0:
            continue

        for obj_id, bbox_xywh in zip(obj_ids, bboxes):
            region_id = f"{sample_id}_{obj_id}"
            crop = _crop_bytes(page_image_bytes, bbox_xywh, page_w, page_h)
            success = len(crop) > 0

            if not success:
                records.append(ImageBenchmarkRecord(
                    sample_id=sample_id,
                    region_id=region_id,
                    bbox=_normalize_bbox_xywh(bbox_xywh),
                    extraction_success=False,
                    fidelity_score=0.0,
                    passed_threshold=False,
                    phash_similarity=0.0,
                    sharpness_crop=0.0,
                    sharpness_original=0.0,
                    sharpness_ratio=0.0,
                    caption_found=False,
                    enrichment_attempted=False,
                    enrichment_populated=False,
                    image_type=None,
                    has_description=False,
                    has_extracted_text=False,
                    has_structured_data=False,
                ))
                continue

            region = _make_region(region_id, crop, bbox_xywh)
            entity = ExtractedEntity(
                region_id=region_id,
                entity_type=EntityType.IMAGE,
                content=ImageContent(
                    crop_bytes=crop,
                    classification_label=None,
                    classification_confidence=None,
                ),
                extractor_name="primary",
            )

            # all_regions is empty in standalone evaluation: caption_found=False for all
            reconstruction = reconstruct_image(entity, region, all_regions=[])
            fidelity = compare_image(reconstruction.content, region, settings.threshold_image)

            enrichment_attempted = enrich_count < enrich_limit
            if enrichment_attempted:
                enrich_count += 1
                entity = enrich_image(entity, context_text="")
            image_content: ImageContent = entity.content
            enrichment_populated = any(
                [
                    image_content.image_type,
                    image_content.description,
                    image_content.extracted_text,
                    image_content.structured_data,
                ]
            )

            sharpness_ratio = (
                min(reconstruction.content.sharpness_crop / reconstruction.content.sharpness_original, 1.0)
                if reconstruction.content.sharpness_original > 0
                else 0.0
            )

            records.append(ImageBenchmarkRecord(
                sample_id=sample_id,
                region_id=region_id,
                bbox=_normalize_bbox_xywh(bbox_xywh),
                extraction_success=True,
                fidelity_score=fidelity.fidelity_score,
                passed_threshold=fidelity.passed_threshold,
                phash_similarity=fidelity.component_scores.get("phash_similarity", 0.0),
                sharpness_crop=reconstruction.content.sharpness_crop,
                sharpness_original=reconstruction.content.sharpness_original,
                sharpness_ratio=sharpness_ratio,
                caption_found=reconstruction.content.caption_found,
                enrichment_attempted=enrichment_attempted,
                enrichment_populated=enrichment_populated,
                image_type=image_content.image_type,
                has_description=bool(image_content.description),
                has_extracted_text=bool(image_content.extracted_text),
                has_structured_data=bool(image_content.structured_data),
            ))

    if not records:
        raise RuntimeError("No annotated regions found in dataset. Check split and dataset path.")

    successful = [r for r in records if r.extraction_success]
    n = len(records)
    ns = len(successful)

    summary = ImageBenchmarkSummary(
        split=split,
        num_pages=frame.shape[0],
        num_regions=n,
        extraction_success_rate=ns / n if n else 0.0,
        mean_fidelity=sum(r.fidelity_score for r in successful) / ns if ns else 0.0,
        pass_rate=sum(1 for r in successful if r.passed_threshold) / ns if ns else 0.0,
        mean_sharpness_crop=sum(r.sharpness_crop for r in successful) / ns if ns else 0.0,
        mean_sharpness_ratio=sum(r.sharpness_ratio for r in successful) / ns if ns else 0.0,
        caption_detection_rate=sum(1 for r in successful if r.caption_found) / ns if ns else 0.0,
        enrichment_attempt_rate=sum(1 for r in successful if r.enrichment_attempted) / ns if ns else 0.0,
        enrichment_success_rate=sum(1 for r in successful if r.enrichment_populated) / ns if ns else 0.0,
        description_coverage=sum(1 for r in successful if r.has_description) / ns if ns else 0.0,
        extracted_text_coverage=sum(1 for r in successful if r.has_extracted_text) / ns if ns else 0.0,
        structured_data_coverage=sum(1 for r in successful if r.has_structured_data) / ns if ns else 0.0,
    )
    return summary, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3b ScanBank image benchmark.")
    parser.add_argument(
        "--dataset-root",
        default="data/raw/scanbank",
        help="Path to the ScanBank dataset root.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enrich-limit", type=int, default=20, help="Maximum number of image regions to send to the semantic enricher.")
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_image_benchmark(
        args.dataset_root,
        split=args.split,
        limit=args.limit,
        enrich_limit=args.enrich_limit,
    )
    payload = {
        "summary": asdict(summary),
        "records": [asdict(r) for r in records],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
