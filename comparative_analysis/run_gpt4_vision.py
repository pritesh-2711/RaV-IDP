"""Baseline: GPT-4.1 direct vision QA (no extraction step).

Sends the raw document image directly to GPT-4.1 with the question.
No intermediate text extraction — the model answers from the image.
This represents the upper bound of "just use GPT-4o/4.1 on everything".

Cost: ~300 calls × gpt-4.1 vision pricing (~$0.01-0.02/call) ≈ $3-6 total.
Checkpoints per-question so the run is resumable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparative_analysis.common import (
    anls_score, load_docvqa_frame, load_json, save_json,
)

_QA_SYSTEM = (
    "You are a precise document question answering assistant. "
    "Look at the document image and answer the question. "
    "Reply with ONLY the answer value — a word, number, name, or short phrase. "
    "Do NOT write a full sentence. Do NOT include the question or any explanation. "
    "If you cannot find the answer, reply with 'unanswerable'."
)


@dataclass
class VisionQARecord:
    question_id: str
    doc_id: str
    question: str
    predicted_answer: str
    gt_answers: list[str]
    anls: float
    extractor: str = "gpt4_vision"
    error: str | None = None


@dataclass
class VisionSummary:
    extractor: str
    num_questions: int
    num_docs: int
    mean_anls: float
    answerable_rate: float
    error_rate: float


def _prepare_image(image_bytes: bytes, max_side: int = 2048) -> tuple[bytes, str]:
    """Resize image to max_side on longest edge and return (png_bytes, mime)."""
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None
    pil = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), PIL.Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _answer_from_image(image_bytes: bytes, question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resized_bytes, mime = _prepare_image(image_bytes)
    img_b64 = base64.b64encode(resized_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        max_tokens=128,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail": "high"}},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            },
        ],
    )
    return response.choices[0].message.content.strip()


def run_gpt4_vision_baseline(
    frame: pd.DataFrame,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> tuple[VisionSummary, list[VisionQARecord]]:
    ckpt_path = checkpoint_dir / "gpt4_vision_qa_records.json"
    completed: list[dict] = load_json(ckpt_path)
    done = {r["question_id"] for r in completed}

    rows = list(frame.itertuples(index=False))
    qa_dicts: list[dict] = list(completed)

    for j, row in enumerate(rows):
        question_id = str(getattr(row, "questionId", getattr(row, "question_id", "")))
        if question_id in done:
            continue

        doc_id = str(getattr(row, "docId", getattr(row, "doc_id", question_id)))
        question = str(row.question)
        gt_answers = list(row.answers) if hasattr(row, "answers") else []

        image_col = row.image
        image_bytes = image_col["bytes"] if isinstance(image_col, dict) else image_col

        try:
            predicted = _answer_from_image(image_bytes, question)
            error = None
        except Exception as exc:
            predicted = ""; error = str(exc)

        rec = dict(
            question_id=question_id, doc_id=doc_id, question=question,
            predicted_answer=predicted, gt_answers=gt_answers,
            anls=anls_score(predicted, gt_answers),
            extractor="gpt4_vision", error=error,
        )
        qa_dicts.append(rec)
        done.add(question_id)

        if verbose:
            print(f"[gpt4_vision] q={j+1}/{len(rows)} anls={rec['anls']:.3f}", flush=True)

        if (j + 1) % 25 == 0:
            save_json(ckpt_path, qa_dicts)

    save_json(ckpt_path, qa_dicts)
    records = [VisionQARecord(**r) for r in qa_dicts]

    n = len(records)
    mean_anls = sum(r.anls for r in records) / n if n else 0.0
    answerable = sum(1 for r in records if r.anls > 0) / n if n else 0.0
    error_rate = sum(1 for r in records if r.error) / n if n else 0.0
    num_docs = len({r.doc_id for r in records})

    summary = VisionSummary(
        extractor="gpt4_vision",
        num_questions=n,
        num_docs=num_docs,
        mean_anls=round(mean_anls, 4),
        answerable_rate=round(answerable, 4),
        error_rate=round(error_rate, 4),
    )
    return summary, records


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-4.1 direct vision QA baseline.")
    ap.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--checkpoint-dir", default="artifacts/baselines/checkpoints")
    ap.add_argument("--output", default="artifacts/baselines/gpt4_vision_300_val.json")
    args = ap.parse_args()

    frame = load_docvqa_frame(args.dataset_root, args.split, args.limit)
    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    summary, records = run_gpt4_vision_baseline(frame, ckpt)

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
