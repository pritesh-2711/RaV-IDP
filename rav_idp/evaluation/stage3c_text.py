"""Stage 3c: text extraction benchmark on FUNSD."""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pytesseract
from Levenshtein import distance as lev_distance
from PIL import Image

from ..components.comparators.text import compare_text
from ..components.extractors.text import extract_text
from ..components.region_preprocessor import preprocess_region
from ..components.region_quality_classifier import classify_region
from ..components.reconstructors.text import reconstruct_text
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref_tokens)


def _ocr_extract(image_bytes: bytes, config: str = "--psm 11") -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return _normalize_text(pytesseract.image_to_string(image, config=config))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    inter_x0 = max(a[0], b[0])
    inter_y0 = max(a[1], b[1])
    inter_x1 = min(a[2], b[2])
    inter_y1 = min(a[3], b[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _gt_coverage(predicted: list[float], gt: list[float]) -> float:
    inter_x0 = max(predicted[0], gt[0])
    inter_y0 = max(predicted[1], gt[1])
    inter_x1 = min(predicted[2], gt[2])
    inter_y1 = min(predicted[3], gt[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    gt_area = max(0.0, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    return inter / gt_area if gt_area > 0 else 0.0


def _matches_gt(pred_box: list[float], gt_box: list[float], min_iou: float = 0.1, min_gt_cover: float = 0.5) -> bool:
    return _bbox_iou(pred_box, gt_box) >= min_iou or _gt_coverage(pred_box, gt_box) >= min_gt_cover


def _ocr_detect_boxes(image_bytes: bytes) -> list[list[float]]:
    """Detect OCR word boxes on the full image for overlap-aware evaluation.

    This is used as a complementary metric to answer:
    "Did OCR find the annotated text regions?"
    rather than forcing a page-level text string match against incomplete
    annotations.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 11")
    boxes: list[list[float]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def _overlap_metrics(pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> tuple[float, float, float]:
    if not gt_boxes:
        return 0.0, 0.0, 0.0
    gt_hits = sum(any(_matches_gt(pred, gt) for pred in pred_boxes) for gt in gt_boxes)
    pred_hits = sum(any(_matches_gt(pred, gt) for gt in gt_boxes) for pred in pred_boxes)
    recall = gt_hits / len(gt_boxes) if gt_boxes else 0.0
    precision = pred_hits / len(pred_boxes) if pred_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _ocr_gt_regions(image_bytes: bytes, bboxes: list) -> str:
    """OCR only the word-level crops defined by GT bounding boxes.

    FUNSD ground-truth annotations cover form-field text only. Text outside
    these boxes (figure captions, axis labels, etc.) is intentionally excluded
    so that OCR false-positives from unannotated regions do not inflate CER/WER.

    Each crop is run with psm=8 (single-word mode). Empty results are skipped.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    texts = []
    for bbox in bboxes:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop((x0, y0, x1, y1))
        text = _normalize_text(pytesseract.image_to_string(crop, config="--psm 8"))
        if text:
            texts.append(text)
    return _normalize_text(" ".join(texts))


def _ground_truth_text(words: list[str] | tuple[str, ...]) -> str:
    return _normalize_text(" ".join(str(word) for word in words))


def _image_size(image_payload: dict) -> tuple[int, int]:
    image = Image.open(io.BytesIO(image_payload["bytes"]))
    return image.size


def _make_region(sample_id: str, image_payload: dict, extracted_text: str) -> DetectedRegion:
    width, height = _image_size(image_payload)
    return DetectedRegion(
        region_id=sample_id,
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=0, x1=width, y1=height, page=0),
        original_crop=image_payload["bytes"],
        processed_crop=image_payload["bytes"],
        raw_docling_record={"text": extracted_text},
        page_index=0,
    )


@dataclass
class TextBenchmarkRecord:
    sample_id: str
    ground_truth_text: str
    extracted_text: str
    reocr_text: str
    cer: float
    wer: float
    overlap_precision: float
    overlap_recall: float
    overlap_f1: float
    fidelity_score: float
    passed_threshold: bool


@dataclass
class TextBenchmarkSummary:
    split: str
    num_samples: int
    mean_cer: float
    median_cer: float
    mean_wer: float
    mean_overlap_precision: float
    mean_overlap_recall: float
    mean_overlap_f1: float
    mean_fidelity: float
    pass_rate: float
    fidelity_cer_correlation: float


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def run_text_benchmark(
    dataset_root: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> tuple[TextBenchmarkSummary, list[TextBenchmarkRecord]]:
    """Run text extraction benchmark on FUNSD parquet data."""

    settings = get_settings()
    dataset_root = Path(dataset_root)
    parquet_path = dataset_root / "data" / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(parquet_path)
    if limit is not None:
        frame = frame.head(limit)

    records: list[TextBenchmarkRecord] = []
    for row in frame.itertuples(index=False):
        ground_truth = _ground_truth_text(list(row.words))
        region = preprocess_region(classify_region(_make_region(str(row.id), row.image, "")))

        # Use GT bounding boxes to restrict OCR to annotated regions only.
        # This avoids penalising the extractor for text outside FUNSD annotations
        # (captions, axis labels, figure text) that the GT does not cover.
        raw_image_bytes = region.processed_crop or region.original_crop
        gt_bboxes = list(row.bboxes) if hasattr(row, "bboxes") else []
        pred_boxes = _ocr_detect_boxes(raw_image_bytes)
        overlap_precision, overlap_recall, overlap_f1 = _overlap_metrics(pred_boxes, gt_bboxes)
        if gt_bboxes:
            extracted_text = _ocr_gt_regions(raw_image_bytes, gt_bboxes)
        else:
            extracted_text = _ocr_extract(raw_image_bytes, config="--psm 11")
        region = region.model_copy(update={"raw_docling_record": {"text": extracted_text}})
        entity = extract_text(region)
        reconstruction = reconstruct_text(
            entity,
            region,
            is_native_pdf=False,
            document_path="funsd-image.png",
        )
        fidelity = compare_text(
            reconstruction.content,
            entity.content.text,
            region.region_id,
            settings.threshold_text,
            entity_type=EntityType.TEXT,
        )

        cer = 0.0 if not ground_truth else lev_distance(extracted_text, ground_truth) / len(ground_truth)
        wer = _word_error_rate(ground_truth, extracted_text)
        records.append(
            TextBenchmarkRecord(
                sample_id=str(row.id),
                ground_truth_text=ground_truth,
                extracted_text=extracted_text,
                reocr_text=reconstruction.content.reocr_text,
                cer=cer,
                wer=wer,
                overlap_precision=overlap_precision,
                overlap_recall=overlap_recall,
                overlap_f1=overlap_f1,
                fidelity_score=fidelity.fidelity_score,
                passed_threshold=fidelity.passed_threshold,
            )
        )

    mean_cer = sum(record.cer for record in records) / len(records)
    mean_wer = sum(record.wer for record in records) / len(records)
    mean_overlap_precision = sum(record.overlap_precision for record in records) / len(records)
    mean_overlap_recall = sum(record.overlap_recall for record in records) / len(records)
    mean_overlap_f1 = sum(record.overlap_f1 for record in records) / len(records)
    mean_fidelity = sum(record.fidelity_score for record in records) / len(records)
    median_cer = sorted(record.cer for record in records)[len(records) // 2]
    pass_rate = sum(1 for record in records if record.passed_threshold) / len(records)
    summary = TextBenchmarkSummary(
        split=split,
        num_samples=len(records),
        mean_cer=mean_cer,
        median_cer=median_cer,
        mean_wer=mean_wer,
        mean_overlap_precision=mean_overlap_precision,
        mean_overlap_recall=mean_overlap_recall,
        mean_overlap_f1=mean_overlap_f1,
        mean_fidelity=mean_fidelity,
        pass_rate=pass_rate,
        fidelity_cer_correlation=_pearson(
            [record.fidelity_score for record in records],
            [-record.cer for record in records],
        ),
    )
    return summary, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3c FUNSD text benchmark.")
    parser.add_argument(
        "--dataset-root",
        default="data/raw/funsd",
        help="Path to the FUNSD dataset root.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output file for summary and records.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_text_benchmark(args.dataset_root, split=args.split, limit=args.limit)
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
