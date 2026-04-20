"""Baseline: Docling alone (no RaV preprocessing, no fidelity gate).

Runs DocumentConverter directly on each DocVQA document and exports
the result as Markdown. No region filtering, dedup, or quality classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparative_analysis.common import (
    BaselineSummary, load_docvqa_frame, run_baseline,
)


_docling_converter = None


def _get_converter():
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True          # force OCR for scanned images
        pipeline_options.do_table_structure = True
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    return _docling_converter


def _extract(pdf_path: str) -> str:
    converter = _get_converter()
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()
    # filter out pure image placeholder lines
    lines = [l for l in md.splitlines() if l.strip() and l.strip() != "<!-- image -->"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--checkpoint-dir", default="artifacts/baselines/checkpoints")
    ap.add_argument("--output", default="artifacts/baselines/docling_300_val.json")
    args = ap.parse_args()

    frame = load_docvqa_frame(args.dataset_root, args.split, args.limit)
    ckpt = Path(args.checkpoint_dir)
    summary, records = run_baseline("docling", _extract, frame, ckpt)

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
