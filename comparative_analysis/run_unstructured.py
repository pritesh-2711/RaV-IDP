"""Baseline: Unstructured.io (open-source, local PDF partitioning).

Install: pip install "unstructured[pdf]"
System deps: apt-get install -y poppler-utils tesseract-ocr
Uses hi_res strategy with OCR for scanned images.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparative_analysis.common import load_docvqa_frame, run_baseline


def _extract(pdf_path: str) -> str:
    from unstructured.partition.pdf import partition_pdf
    # hi_res uses layout detection + OCR; best quality for scanned docs
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
    )
    parts = []
    for el in elements:
        text = str(el).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--checkpoint-dir", default="artifacts/baselines/checkpoints")
    ap.add_argument("--output", default="artifacts/baselines/unstructured_300_val.json")
    args = ap.parse_args()

    frame = load_docvqa_frame(args.dataset_root, args.split, args.limit)
    ckpt = Path(args.checkpoint_dir)
    summary, records = run_baseline("unstructured", _extract, frame, ckpt)

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
