"""Baseline: Marker (Surya OCR + layout detection → Markdown).

Install: pip install marker-pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparative_analysis.common import load_docvqa_frame, run_baseline

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        _converter = PdfConverter(artifact_dict=create_model_dict())
    return _converter


def _extract(pdf_path: str) -> str:
    converter = _get_converter()
    rendered = converter(pdf_path)
    return rendered.markdown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--checkpoint-dir", default="artifacts/baselines/checkpoints")
    ap.add_argument("--output", default="artifacts/baselines/marker_300_val.json")
    args = ap.parse_args()

    frame = load_docvqa_frame(args.dataset_root, args.split, args.limit)
    ckpt = Path(args.checkpoint_dir)
    summary, records = run_baseline("marker", _extract, frame, ckpt)

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
