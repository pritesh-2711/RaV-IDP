"""Focused regression test for TATR span-merging + GPT fallback on 3 known mismatch samples.

Samples chosen because the previous benchmark revealed clear failure modes:
  549107 — gt_rows=1→31  pred_rows=31  col mismatch (GT=9, pred=6)
  549181 — gt_rows=1→48  pred_rows=60  TATR over-segments by 12
  549302 — gt_rows=0→14  pred_rows=12  span-merge helped (was 14→12 after merge)

Run from experiment/:
  ../.venv/bin/python test_table_fixes.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from rav_idp.evaluation.stage3a_tables import (
    _TATR_MIN_IMAGE_DIM,
    _TATR_THRESHOLD,
    _cer,
    _dataframe_signature,
    _derive_ground_truth,
    _docling_table_record,
    _gpt_table_record,
    _iter_annotations,
    _load_image_bytes_batch,
    _make_region,
    _merge_spanned_rows,
    _pad_table_image,
    _resolve_tatr_source,
    _save_mismatch_visual,
    _tatr_col_count,
    _tatr_table_record,
    _upscale_image,
    TableBenchmarkRecord,
)
from rav_idp.components.comparators.table import compare_table
from rav_idp.components.extractors.table import extract_table
from rav_idp.components.reconstructors.table import reconstruct_table
from rav_idp.components.region_preprocessor import preprocess_region
from rav_idp.components.region_quality_classifier import classify_region
from rav_idp.config import get_settings
from rav_idp.models import QualityClass

TARGET_IDS = {"549107", "549181", "549302"}

BEFORE = {
    "549107": {"gt_rows": 1, "pred_rows": 31, "gt_cols": 9, "pred_cols": 6},
    "549181": {"gt_rows": 1, "pred_rows": 60, "gt_cols": 9, "pred_cols": 9},
    "549302": {"gt_rows": 0, "pred_rows": 14, "gt_cols": 7, "pred_cols": 7},
}

DATASET_ROOT = Path("data/raw/pubtabnet")
ARCHIVE = DATASET_ROOT / "pubtabnet.tar.gz"
OUT_DIR = Path("artifacts/stage3a_fix_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _tatr_raw_bands(
    image_bytes: bytes,
    model: TableTransformerForObjectDetection,
    processor: AutoImageProcessor,
) -> tuple[list[tuple[float, list[float]]], list[tuple[float, list[float]]]]:
    """Return (row_bands, col_bands) as (y_center, [x0,y0,x1,y1]) lists from raw TATR detections."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _pad_table_image(image)
    image = _upscale_image(image, _TATR_MIN_IMAGE_DIM)
    w, h = image.size

    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs, threshold=_TATR_THRESHOLD, target_sizes=target_sizes
    )[0]

    label_map = model.config.id2label
    rows: list[tuple[float, list[float]]] = []
    cols: list[tuple[float, list[float]]] = []
    spanning: list[list[float]] = []

    for label_id, box in zip(results["labels"].cpu().tolist(), results["boxes"].cpu().tolist()):
        name = label_map.get(label_id, "").lower()
        x0, y0, x1, y1 = box
        if "spanning" in name:
            spanning.append([x0, y0, x1, y1])
        elif "row" in name and "header" not in name and "projected" not in name:
            rows.append(((y0 + y1) / 2.0, [x0, y0, x1, y1]))
        elif "column" in name and "header" not in name:
            cols.append(((x0 + x1) / 2.0, [x0, y0, x1, y1]))

    rows.sort(key=lambda r: r[0])
    cols.sort(key=lambda c: c[0])
    rows = _merge_spanned_rows(rows, spanning)

    # Scale band coords back to original image size (undo pad+upscale).
    # For overlay purposes we just use the padded/upscaled coords —
    # the visual is annotated on the original image, so rescale linearly.
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    sx = orig.width / w
    sy = orig.height / h

    def _scale_band(b: tuple[float, list[float]]) -> tuple[float, list[float]]:
        _, (bx0, by0, bx1, by1) = b
        return (b[0] * sy, [bx0 * sx, by0 * sy, bx1 * sx, by1 * sy])

    return [_scale_band(r) for r in rows], [_scale_band(c) for c in cols]


def main() -> None:
    settings = get_settings()

    print("Loading TATR model …")
    src = _resolve_tatr_source()
    local_only = src != "microsoft/table-transformer-structure-recognition"
    tatr_processor = AutoImageProcessor.from_pretrained(src, local_files_only=local_only)
    tatr_model = TableTransformerForObjectDetection.from_pretrained(src, local_files_only=local_only)
    tatr_model.eval()

    print("Scanning val annotations for target samples …")
    all_annotations = _iter_annotations(DATASET_ROOT, ARCHIVE, split="val", limit=None)
    target_annotations = [
        a for a in all_annotations
        if str(a.get("imgid", a.get("filename", ""))) in TARGET_IDS
    ]
    if not target_annotations:
        raise RuntimeError("None of the target samples were found in the val split.")
    print(f"Found {len(target_annotations)} / {len(TARGET_IDS)} targets.\n")

    filenames = [str(a["filename"]) for a in target_annotations]
    image_bytes_map = _load_image_bytes_batch(ARCHIVE, "val", filenames)

    for annotation in target_annotations:
        gt = _derive_ground_truth(annotation)
        sample_id = str(annotation.get("imgid", gt.filename))
        image_bytes = image_bytes_map[gt.filename]

        region = preprocess_region(classify_region(_make_region(sample_id, image_bytes)))
        crop_bytes = region.processed_crop or image_bytes
        is_degraded = region.quality_class == QualityClass.SCANNED_DEGRADED

        print(f"{'='*60}")
        print(f"Sample : {sample_id}  ({gt.filename})")
        print(f"Quality: {region.quality_class}  degraded={is_degraded}")

        if is_degraded:
            print("  → routing to GPT-4o fallback")
            table_record = _gpt_table_record(crop_bytes)
            detected_cols = _tatr_col_count(table_record) if table_record else None
            source = "GPT"
            tatr_row_bands, tatr_col_bands = None, None
            if not table_record:
                print("  → GPT unavailable / failed, falling back to TATR")
                tatr_record = _tatr_table_record(crop_bytes)
                detected_cols = _tatr_col_count(tatr_record)
                table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)
                source = "TATR (GPT fallback failed)"
                tatr_row_bands, tatr_col_bands = _tatr_raw_bands(crop_bytes, tatr_model, tatr_processor)
        else:
            tatr_record = _tatr_table_record(crop_bytes)
            detected_cols = _tatr_col_count(tatr_record)
            table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)
            source = "TATR + span-merge"
            tatr_row_bands, tatr_col_bands = _tatr_raw_bands(crop_bytes, tatr_model, tatr_processor)

        region = region.model_copy(update={"raw_docling_record": table_record})
        entity = extract_table(region)
        reconstruction = reconstruct_table(entity, region)
        fidelity = compare_table(
            reconstruction.content,
            region,
            settings.threshold_table,
            skip_visual=True,
            detected_col_count=detected_cols,
        )

        pred_rows, pred_cols, pred_headers, pred_cells = _dataframe_signature(
            entity.content.dataframe_json
        )
        combined_gt = gt.headers + gt.cell_texts
        combined_pred = pred_headers + pred_cells
        cer = _cer(combined_gt, combined_pred)

        before = BEFORE.get(sample_id, {})
        row_ok = gt.row_count == pred_rows
        col_ok = gt.col_count == pred_cols

        print(f"  Source    : {source}")
        print(f"  GT        : rows={gt.row_count}  cols={gt.col_count}")
        print(f"  BEFORE    : gt_rows={before.get('gt_rows','?')}  pred_rows={before.get('pred_rows','?')}  "
              f"gt_cols={before.get('gt_cols','?')}  pred_cols={before.get('pred_cols','?')}")
        print(f"  AFTER     : rows={pred_rows}  cols={pred_cols}")
        print(f"  row_match={row_ok}  col_match={col_ok}")
        print(f"  CER={cer:.3f}  fidelity={fidelity.fidelity_score:.3f}  passed={fidelity.passed_threshold}")
        if tatr_row_bands:
            print(f"  TATR bands: {len(tatr_row_bands)} row bands, {len(tatr_col_bands)} col bands")

        rec = TableBenchmarkRecord(
            sample_id=sample_id,
            filename=gt.filename,
            ground_truth_rows=gt.row_count,
            predicted_rows=pred_rows,
            ground_truth_cols=gt.col_count,
            predicted_cols=pred_cols,
            ground_truth_nonempty_cells=len(gt.cell_texts),
            predicted_nonempty_cells=len(pred_cells),
            row_match=row_ok,
            col_match=col_ok,
            cell_text_cer=cer,
            fidelity_score=fidelity.fidelity_score,
            passed_threshold=fidelity.passed_threshold,
        )
        _save_mismatch_visual(
            rec,
            image_bytes,
            entity.content.dataframe_json,
            OUT_DIR,
            tatr_row_bands=tatr_row_bands,
            tatr_col_bands=tatr_col_bands,
        )
        print(f"  Saved     : {OUT_DIR}/{sample_id}.png")

    print(f"\nDone. Grid-overlay comparison images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
