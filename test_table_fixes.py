"""Focused regression test for TATR span-merging + GPT fallback on 3 known mismatch samples.

Samples chosen because the previous benchmark revealed clear failure modes:
  549107 — gt_rows=1  pred_rows=31  (TATR over-segments merged-cell table)
  549181 — gt_rows=1  pred_rows=60  (same, plus degraded image quality)
  549302 — gt_rows=0  pred_rows=14  (header-only table; TATR detects all bands as data rows)

Run from experiment/:
  ../.venv/bin/python test_table_fixes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from experiment/ directory
sys.path.insert(0, str(Path(__file__).parent))

from rav_idp.evaluation.stage3a_tables import (
    _iter_annotations,
    _load_image_bytes_batch,
    _derive_ground_truth,
    _make_region,
    _tatr_table_record,
    _tatr_col_count,
    _docling_table_record,
    _gpt_table_record,
    _dataframe_signature,
    _cer,
    _save_mismatch_visual,
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

# Results from the previous benchmark run (before the fixes).
BEFORE = {
    "549107": {"pred_rows": 31, "pred_cols": 6},
    "549181": {"pred_rows": 60, "pred_cols": 9},
    "549302": {"pred_rows": 14, "pred_cols": 7},
}

DATASET_ROOT = Path("data/raw/pubtabnet")
ARCHIVE = DATASET_ROOT / "pubtabnet.tar.gz"
OUT_DIR = Path("artifacts/stage3a_fix_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = get_settings()

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
            if not table_record:
                print("  → GPT unavailable / failed, falling back to TATR")
                tatr_record = _tatr_table_record(crop_bytes)
                detected_cols = _tatr_col_count(tatr_record)
                table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)
                source = "TATR (GPT fallback failed)"
        else:
            tatr_record = _tatr_table_record(crop_bytes)
            detected_cols = _tatr_col_count(tatr_record)
            table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)
            source = "TATR + span-merge"

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
        row_delta = pred_rows - before.get("pred_rows", pred_rows)
        col_delta = pred_cols - before.get("pred_cols", pred_cols)

        print(f"  Source : {source}")
        print(f"  GT     : rows={gt.row_count}  cols={gt.col_count}")
        print(
            f"  BEFORE : rows={before.get('pred_rows', '?')}  cols={before.get('pred_cols', '?')}"
        )
        print(
            f"  AFTER  : rows={pred_rows}  cols={pred_cols}"
            f"  (Δrows={row_delta:+d}  Δcols={col_delta:+d})"
        )
        print(f"  CER={cer:.3f}  fidelity={fidelity.fidelity_score:.3f}  passed={fidelity.passed_threshold}")
        row_ok = gt.row_count == pred_rows
        col_ok = gt.col_count == pred_cols
        print(f"  row_match={row_ok}  col_match={col_ok}")

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
        _save_mismatch_visual(rec, image_bytes, entity.content.dataframe_json, OUT_DIR)
        print(f"  Saved  : {OUT_DIR}/{sample_id}.png")

    print(f"\nDone. Comparison images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
