"""Stage 6: end-to-end document QA benchmark on DocVQA.

Runs the full RaV-IDP pipeline on each DocVQA document, builds a context
string from extracted entities, then answers each question with GPT-4o.
Measures ANLS (Average Normalized Levenshtein Similarity) against GT answers.

Dataset: DocVQA — requires registration at https://www.docvqa.org/
         or the HuggingFace mirror lmms-lab/DocVQA (gated, login required).
         Expected parquet layout: data/{split}-*.parquet
         Expected columns: questionId, question, answers, image, docId

ANLS formula (per DocVQA paper):
  NLS(pred, gt) = edit_distance(pred, gt) / max(len(pred), len(gt))
  ANLS_per_pair = 1 - NLS  if NLS < 0.5  else  0
  ANLS_per_question = max(ANLS_per_pair over all GT answers)
  Final ANLS = mean over all questions

OPENAI_API_KEY is required for both the QA step and image enrichment.
Set RAV_SKIP_ENRICHMENT=1 to disable image enrichment and reduce API calls.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import pandas as pd
from Levenshtein import distance as lev_distance

from ..config import get_settings
from ..models import EntityType
from ..pipeline import RaVIDPPipeline


# ---------------------------------------------------------------------------
# ANLS helpers
# ---------------------------------------------------------------------------

def _nls(pred: str, gt: str) -> float:
    pred, gt = pred.strip().lower(), gt.strip().lower()
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    return lev_distance(pred, gt) / max_len


def _anls_score(pred: str, gt_answers: list[str]) -> float:
    if not gt_answers:
        return 0.0
    best = max(1 - _nls(pred, gt) if _nls(pred, gt) < 0.5 else 0.0 for gt in gt_answers)
    return best


# ---------------------------------------------------------------------------
# Context building from pipeline entity records
# ---------------------------------------------------------------------------

def _build_context(entity_records: list) -> str:
    """Build a flat text context string from extracted entity records.

    Text entities contribute their raw text.
    Table entities contribute their markdown representation.
    Image entities contribute their description if the enricher populated it.
    Entities are ordered by page then by vertical position.
    """
    ordered = sorted(
        entity_records,
        key=lambda r: (r.page_index, r.bbox.y0),
    )
    parts: list[str] = []
    for record in ordered:
        if record.entity_type == EntityType.TEXT:
            text = getattr(record.content, "text", "").strip()
            if text:
                parts.append(text)
        elif record.entity_type == EntityType.TABLE:
            md = getattr(record.content, "markdown", "").strip()
            if md:
                parts.append(md)
        elif record.entity_type == EntityType.IMAGE:
            desc = getattr(record.content, "description", None)
            extracted = getattr(record.content, "extracted_text", None)
            if extracted:
                parts.append(extracted)
            elif desc:
                parts.append(f"[Figure: {desc}]")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# QA via OpenAI
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise document question answering assistant. "
    "Answer the question using only information from the provided document context. "
    "Be concise. If the answer is not in the context, reply with 'unanswerable'."
)


def _answer_question(context: str, question: str, settings) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=128,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Document ingestion — convert image to single-page PDF for the pipeline
# ---------------------------------------------------------------------------

def _image_bytes_to_pdf(image_bytes: bytes) -> bytes:
    """Wrap a raster image in a single-page PDF so the pipeline can process it."""
    pil_buf = io.BytesIO(image_bytes)
    import PIL.Image
    pil_img = PIL.Image.open(pil_buf).convert("RGB")
    w, h = pil_img.size
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    page.insert_image(page.rect, stream=img_buf.getvalue())
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Records and summary
# ---------------------------------------------------------------------------

@dataclass
class QARecord:
    question_id: str
    doc_id: str
    question: str
    predicted_answer: str
    gt_answers: list[str]
    anls: float
    pipeline_error: str | None


@dataclass
class Stage6Summary:
    split: str
    num_questions: int
    num_docs: int
    mean_anls: float
    answerable_rate: float   # fraction where anls > 0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_endtoend_benchmark(
    dataset_root: str | Path,
    split: str = "val",
    limit: int | None = None,
) -> tuple[Stage6Summary, list[QARecord]]:
    """Run end-to-end DocVQA benchmark."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for stage 6. "
            "Set it in .env or as an environment variable."
        )

    dataset_root = Path(dataset_root)
    parquet_candidates = list(dataset_root.glob(f"data/{split}*.parquet"))
    if not parquet_candidates:
        parquet_candidates = list(dataset_root.glob(f"{split}*.parquet"))
    if not parquet_candidates:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' under {dataset_root}. "
            "Download DocVQA from https://www.docvqa.org/ or via HuggingFace."
        )

    frame = pd.read_parquet(parquet_candidates[0])
    if limit is not None:
        frame = frame.head(limit)

    pipeline = RaVIDPPipeline()
    # cache processed docs to avoid re-running the pipeline for the same doc
    doc_context_cache: dict[str, str] = {}

    qa_records: list[QARecord] = []

    for row in frame.itertuples(index=False):
        question_id = str(getattr(row, "questionId", getattr(row, "question_id", "")))
        doc_id = str(getattr(row, "docId", getattr(row, "doc_id", question_id)))
        question = str(row.question)
        gt_answers = list(row.answers) if hasattr(row, "answers") else []

        # run pipeline once per doc, cache context
        if doc_id not in doc_context_cache:
            image_col = row.image
            image_bytes = image_col["bytes"] if isinstance(image_col, dict) else image_col

            try:
                pdf_bytes = _image_bytes_to_pdf(image_bytes)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                entity_records = pipeline.run(tmp_path)
                context = _build_context(entity_records)
                doc_context_cache[doc_id] = context
            except Exception as exc:
                doc_context_cache[doc_id] = ""
                qa_records.append(QARecord(
                    question_id=question_id,
                    doc_id=doc_id,
                    question=question,
                    predicted_answer="",
                    gt_answers=gt_answers,
                    anls=0.0,
                    pipeline_error=str(exc),
                ))
                continue
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        context = doc_context_cache[doc_id]
        if not context:
            qa_records.append(QARecord(
                question_id=question_id,
                doc_id=doc_id,
                question=question,
                predicted_answer="",
                gt_answers=gt_answers,
                anls=0.0,
                pipeline_error="empty context — pipeline produced no text",
            ))
            continue

        try:
            predicted = _answer_question(context, question, settings)
            error = None
        except Exception as exc:
            predicted = ""
            error = str(exc)

        anls = _anls_score(predicted, gt_answers)
        qa_records.append(QARecord(
            question_id=question_id,
            doc_id=doc_id,
            question=question,
            predicted_answer=predicted,
            gt_answers=gt_answers,
            anls=anls,
            pipeline_error=error,
        ))

    if not qa_records:
        raise RuntimeError("No QA records produced. Check dataset path and split.")

    n = len(qa_records)
    mean_anls = sum(r.anls for r in qa_records) / n
    answerable = sum(1 for r in qa_records if r.anls > 0) / n
    num_docs = len(doc_context_cache)

    summary = Stage6Summary(
        split=split,
        num_questions=n,
        num_docs=num_docs,
        mean_anls=mean_anls,
        answerable_rate=answerable,
    )
    return summary, qa_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 6 DocVQA end-to-end benchmark.")
    parser.add_argument(
        "--dataset-root",
        default="data/raw/docvqa",
        help="Path to DocVQA dataset root.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_endtoend_benchmark(
        dataset_root=args.dataset_root,
        split=args.split,
        limit=args.limit,
    )
    payload = {
        "summary": asdict(summary),
        "records": [asdict(r) for r in records],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
