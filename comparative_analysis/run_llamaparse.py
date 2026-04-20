"""Baseline: LlamaParse (cloud API, advanced document parsing).

Install: pip install llama-parse
Requires: LLAMA_CLOUD_API_KEY in .env or environment
Free key: https://cloud.llamaindex.ai (sign up → API Keys)

LlamaParse parses PDFs server-side using a proprietary pipeline that
understands tables, figures, and complex layouts. Returns Markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from comparative_analysis.common import load_docvqa_frame, run_baseline


def _extract(pdf_path: str) -> str:
    from llama_parse import LlamaParse
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY not set. Get a free key at https://cloud.llamaindex.ai")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=False,
    )
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--checkpoint-dir", default="artifacts/baselines/checkpoints")
    ap.add_argument("--output", default="artifacts/baselines/llamaparse_300_val.json")
    args = ap.parse_args()

    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("ERROR: LLAMA_CLOUD_API_KEY not set.")
        print("Get a free key at https://cloud.llamaindex.ai and add to experiment/.env:")
        print("  LLAMA_CLOUD_API_KEY=llx-...")
        return 1

    frame = load_docvqa_frame(args.dataset_root, args.split, args.limit)
    ckpt = Path(args.checkpoint_dir)
    summary, records = run_baseline("llamaparse", _extract, frame, ckpt)

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
