"""Stage 2: layout-driven benchmark on a DocLayNet page subset."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

from ..components.layout_detector import detect_layout
from ..components.page_renderer import render_document_pages
from ..models import BoundingBox, DetectedRegion, EntityType

DOCLAYNET_CORE_ZIP = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
LABEL_MAP = {
    "Caption": EntityType.TEXT,
    "Footnote": EntityType.TEXT,
    "Formula": EntityType.FORMULA,
    "List-item": EntityType.TEXT,
    "Page-footer": EntityType.TEXT,
    "Page-header": EntityType.TEXT,
    "Picture": EntityType.IMAGE,
    "Section-header": EntityType.TEXT,
    "Table": EntityType.TABLE,
    "Text": EntityType.TEXT,
    "Title": EntityType.TEXT,
}


@dataclass
class LayoutPageResult:
    image_id: int
    file_name: str
    predicted_counts: dict[str, int]
    ground_truth_counts: dict[str, int]
    matched_iou_mean: float


@dataclass
class LayoutClassSummary:
    label: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float


@dataclass
class LayoutSummary:
    num_pages: int
    iou_threshold: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_class: list[LayoutClassSummary]


def _xywh_to_xyxy(bbox: list[float], page: int = 0) -> BoundingBox:
    x, y, w, h = bbox
    return BoundingBox(x0=x, y0=y, x1=x + w, y1=y + h, page=page)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    inter_x0 = max(a.x0, b.x0)
    inter_y0 = max(a.y0, b.y0)
    inter_x1 = min(a.x1, b.x1)
    inter_y1 = min(a.y1, b.y1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(0.0, (a.x1 - a.x0) * (a.y1 - a.y0))
    area_b = max(0.0, (b.x1 - b.x0) * (b.y1 - b.y0))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _group_ground_truth(annotation: dict, image_id: int) -> list[tuple[EntityType, BoundingBox]]:
    grouped: list[tuple[EntityType, BoundingBox]] = []
    for obj in annotation["objects"]:
        label = LABEL_MAP[obj["category_name"]]
        grouped.append((label, _xywh_to_xyxy(obj["bbox"], page=0)))
    return grouped


def _group_predictions(regions: list[DetectedRegion]) -> list[tuple[EntityType, BoundingBox]]:
    return [(region.entity_type, region.bbox) for region in regions]


def _match_class(
    predictions: list[BoundingBox],
    ground_truth: list[BoundingBox],
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    used_predictions: set[int] = set()
    matched_ious: list[float] = []
    tp = 0
    for gt_box in ground_truth:
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred_box in enumerate(predictions):
            if pred_idx in used_predictions:
                continue
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
        if best_idx is not None and best_iou >= iou_threshold:
            used_predictions.add(best_idx)
            matched_ious.append(best_iou)
            tp += 1

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn, matched_ious


def _load_doclaynet_subset(split: str, limit: int) -> list[dict]:
    with RemoteZip(DOCLAYNET_CORE_ZIP) as archive:
        with archive.open(f"COCO/{split}.json") as handle:
            payload = json.load(handle)

        images = payload["images"][:limit]
        image_ids = {image["id"] for image in images}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        categories = {category["id"]: category["name"] for category in payload["categories"]}

        for annotation in payload["annotations"]:
            if annotation["image_id"] in image_ids:
                annotation = dict(annotation)
                annotation["category_name"] = categories[annotation["category_id"]]
                annotations_by_image[annotation["image_id"]].append(annotation)

        subset: list[dict] = []
        for image in images:
            with archive.open(f"PNG/{image['file_name']}") as image_handle:
                image_bytes = image_handle.read()
            subset.append(
                {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "image_bytes": image_bytes,
                    "objects": annotations_by_image[image["id"]],
                }
            )
    return subset


def _write_temp_png(image_bytes: bytes, file_name: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="rav_doclaynet_"))
    target = temp_dir / file_name
    target.write_bytes(image_bytes)
    return target


def run_layout_benchmark(
    split: str = "test",
    limit: int = 10,
    iou_threshold: float = 0.5,
) -> tuple[LayoutSummary, list[LayoutPageResult]]:
    subset = _load_doclaynet_subset(split, limit)
    per_class_counts: dict[str, dict[str, int | list[float]]] = {
        entity.value: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for entity in (EntityType.TEXT, EntityType.TABLE, EntityType.IMAGE, EntityType.FORMULA)
    }
    page_results: list[LayoutPageResult] = []

    for item in subset:
        image_path = _write_temp_png(item["image_bytes"], item["file_name"])
        pages = render_document_pages(image_path)
        predictions = detect_layout(image_path, pages)

        grouped_gt = _group_ground_truth(item, item["image_id"])
        grouped_pred = _group_predictions(predictions)

        predicted_counts = defaultdict(int)
        ground_truth_counts = defaultdict(int)
        page_ious: list[float] = []

        for entity_type, _ in grouped_pred:
            predicted_counts[entity_type.value] += 1
        for entity_type, _ in grouped_gt:
            ground_truth_counts[entity_type.value] += 1

        for entity in (EntityType.TEXT, EntityType.TABLE, EntityType.IMAGE, EntityType.FORMULA):
            pred_boxes = [bbox for label, bbox in grouped_pred if label == entity]
            gt_boxes = [bbox for label, bbox in grouped_gt if label == entity]
            tp, fp, fn, matched_ious = _match_class(pred_boxes, gt_boxes, iou_threshold)
            per_class_counts[entity.value]["tp"] += tp
            per_class_counts[entity.value]["fp"] += fp
            per_class_counts[entity.value]["fn"] += fn
            per_class_counts[entity.value]["ious"].extend(matched_ious)
            page_ious.extend(matched_ious)

        page_results.append(
            LayoutPageResult(
                image_id=item["image_id"],
                file_name=item["file_name"],
                predicted_counts=dict(predicted_counts),
                ground_truth_counts=dict(ground_truth_counts),
                matched_iou_mean=sum(page_ious) / len(page_ious) if page_ious else 0.0,
            )
        )

    summaries: list[LayoutClassSummary] = []
    total_tp = total_fp = total_fn = 0
    for label, counts in per_class_counts.items():
        tp = int(counts["tp"])
        fp = int(counts["fp"])
        fn = int(counts["fn"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        ious = counts["ious"]
        summaries.append(
            LayoutClassSummary(
                label=label,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                mean_matched_iou=(sum(ious) / len(ious)) if ious else 0.0,
            )
        )

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro_f1 = sum(summary.f1 for summary in summaries) / len(summaries)

    return (
        LayoutSummary(
            num_pages=len(subset),
            iou_threshold=iou_threshold,
            macro_f1=macro_f1,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            per_class=summaries,
        ),
        page_results,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 2 DocLayNet layout benchmark.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, page_results = run_layout_benchmark(
        split=args.split,
        limit=args.limit,
        iou_threshold=args.iou_threshold,
    )
    payload = {
        "summary": asdict(summary),
        "pages": [asdict(page) for page in page_results],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
