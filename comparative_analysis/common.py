"""Shared utilities for DocVQA comparative baselines."""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import pandas as pd
from dotenv import load_dotenv
from Levenshtein import distance as lev_distance

load_dotenv()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DOCVQA_DEFAULT = "data/raw/docvqa/DocVQA"


def load_docvqa_frame(dataset_root: str = DOCVQA_DEFAULT, split: str = "val", limit: int = 300) -> pd.DataFrame:
    root = Path(dataset_root)
    candidates = sorted(root.glob(f"{split}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files for split '{split}' under {root}")
    df = pd.read_parquet(candidates[0])
    return df.head(limit)


def image_bytes_to_pdf(image_bytes: bytes) -> bytes:
    """Wrap a raster image in a standard A4 PDF (avoids giant raster sizes)."""
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None
    pil_img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w_px, h_px = pil_img.size
    page_w, page_h = (842, 595) if w_px > h_px else (595, 842)
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


def write_tmp_pdf(image_bytes: bytes) -> str:
    """Write image as PDF to a temp file; caller must os.unlink() it."""
    pdf_bytes = image_bytes_to_pdf(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        return tmp.name


# ---------------------------------------------------------------------------
# ANLS
# ---------------------------------------------------------------------------

def nls(pred: str, gt: str) -> float:
    pred, gt = pred.strip().lower(), gt.strip().lower()
    denom = max(len(pred), len(gt))
    return 0.0 if denom == 0 else lev_distance(pred, gt) / denom


def anls_score(pred: str, gt_answers: list[str]) -> float:
    if not gt_answers:
        return 0.0
    return max((1 - n) if (n := nls(pred, gt)) < 0.5 else 0.0 for gt in gt_answers)


# ---------------------------------------------------------------------------
# QA via gpt-4.1-mini  (text context → answer)
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are a precise document question answering assistant. "
    "Answer the question using only information from the provided document context. "
    "Give a short, direct answer. If the answer is not in the context, reply with 'unanswerable'."
)


def answer_question(context: str, question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_QA_MODEL", "gpt-4.1-mini"),
        max_tokens=128,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {} if not str(path).endswith("_qa.json") else []


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Benchmark runner (used by each extractor script)
# ---------------------------------------------------------------------------

@dataclass
class QARecord:
    question_id: str
    doc_id: str
    question: str
    predicted_answer: str
    gt_answers: list[str]
    anls: float
    extractor: str
    error: str | None = None


@dataclass
class BaselineSummary:
    extractor: str
    num_questions: int
    num_docs: int
    mean_anls: float
    answerable_rate: float
    error_rate: float


def run_baseline(
    extractor_name: str,
    extract_context_fn,          # fn(pdf_path: str) -> str
    frame: pd.DataFrame,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> tuple[BaselineSummary, list[QARecord]]:
    """Generic runner: process docs once, answer all questions, checkpoint after each doc/QA batch.

    extract_context_fn should raise on hard failure; empty string is an allowed
    result (treated as empty-context error).
    """
    doc_ckpt = checkpoint_dir / f"{extractor_name}_doc_contexts.json"
    qa_ckpt  = checkpoint_dir / f"{extractor_name}_qa_records.json"

    doc_contexts: dict[str, str] = load_json(doc_ckpt)
    completed_qa: list[dict] = load_json(qa_ckpt)
    done_keys = {(r["question_id"],) for r in completed_qa}

    rows = list(frame.itertuples(index=False))

    # --- pass 1: extract doc contexts ---
    for i, row in enumerate(rows):
        doc_id = str(getattr(row, "docId", getattr(row, "doc_id", "")))
        if doc_id in doc_contexts:
            continue
        image_col = row.image
        image_bytes = image_col["bytes"] if isinstance(image_col, dict) else image_col
        tmp_path = None
        try:
            tmp_path = write_tmp_pdf(image_bytes)
            context = extract_context_fn(tmp_path)
            doc_contexts[doc_id] = context or ""
        except Exception as exc:
            doc_contexts[doc_id] = f"__ERROR__:{exc}"
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except Exception: pass
        save_json(doc_ckpt, doc_contexts)
        if verbose:
            status = "ok" if doc_contexts[doc_id] and not doc_contexts[doc_id].startswith("__ERROR__") else "err"
            print(f"[{extractor_name}] doc {i+1}/{len(rows)} {doc_id} [{status}]", flush=True)

    # --- pass 2: answer questions ---
    qa_dicts: list[dict] = list(completed_qa)
    for j, row in enumerate(rows):
        question_id = str(getattr(row, "questionId", getattr(row, "question_id", "")))
        if (question_id,) in done_keys:
            continue
        doc_id = str(getattr(row, "docId", getattr(row, "doc_id", question_id)))
        question = str(row.question)
        gt_answers = list(row.answers) if hasattr(row, "answers") else []
        context = doc_contexts.get(doc_id, "")

        if not context or context.startswith("__ERROR__"):
            error = context.replace("__ERROR__:", "") if context.startswith("__ERROR__") else "empty context"
            rec = dict(question_id=question_id, doc_id=doc_id, question=question,
                       predicted_answer="", gt_answers=gt_answers, anls=0.0,
                       extractor=extractor_name, error=error)
        else:
            try:
                predicted = answer_question(context, question)
                error = None
            except Exception as exc:
                predicted = ""; error = str(exc)
            rec = dict(question_id=question_id, doc_id=doc_id, question=question,
                       predicted_answer=predicted, gt_answers=gt_answers,
                       anls=anls_score(predicted, gt_answers),
                       extractor=extractor_name, error=error)

        qa_dicts.append(rec)
        done_keys.add((question_id,))
        if (j + 1) % 50 == 0:
            save_json(qa_ckpt, qa_dicts)
            if verbose:
                print(f"[{extractor_name}] qa checkpoint at q={j+1}", flush=True)

    save_json(qa_ckpt, qa_dicts)
    records = [QARecord(**r) for r in qa_dicts]

    n = len(records)
    mean_anls = sum(r.anls for r in records) / n if n else 0.0
    answerable = sum(1 for r in records if r.anls > 0) / n if n else 0.0
    error_rate = sum(1 for r in records if r.error) / n if n else 0.0
    num_docs = len({r.doc_id for r in records})

    summary = BaselineSummary(
        extractor=extractor_name,
        num_questions=n,
        num_docs=num_docs,
        mean_anls=round(mean_anls, 4),
        answerable_rate=round(answerable, 4),
        error_rate=round(error_rate, 4),
    )
    return summary, records
