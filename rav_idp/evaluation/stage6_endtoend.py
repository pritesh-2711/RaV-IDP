"""Stage 6: end-to-end document QA benchmark on DocVQA.

Runs the RaV-IDP pipeline on each DocVQA document, builds a context string
from extracted entities, then answers each question with GPT-4o.
Measures ANLS (Average Normalized Levenshtein Similarity) against GT answers.

Three pipeline modes are compared in --mode ablation:
  no_rav    — primary extractor only, no fidelity gate, no fallback
  gate_only — fidelity gate active: low-confidence entities excluded from context
  full      — complete pipeline (gate + GPT-4o fallback for low-confidence entities)

Ablation optimisation: each document is processed ONCE using the full pipeline.
Per-mode contexts are then derived from the stored trace records, avoiding
3× redundant layout/OCR/extraction work.

Dataset: DocVQA — HuggingFace mirror lmms-lab/DocVQA (gated, login required).
  huggingface-cli login
  huggingface-cli download lmms-lab/DocVQA --repo-type dataset \\
      --local-dir data/raw/docvqa
Expected parquet columns: questionId, question, answers, image, docId

ANLS formula (per DocVQA paper):
  NLS(pred, gt) = edit_distance(pred, gt) / max(len(pred), len(gt))
  ANLS_pair    = 1 - NLS  if NLS < 0.5  else  0
  ANLS_q       = max(ANLS_pair over all GT answers for this question)
  Final ANLS   = mean(ANLS_q over all questions)

Model split:
  Extraction / fallback  — settings.openai_model    (default gpt-4.1, vision capable)
  QA answering           — settings.openai_qa_model  (default gpt-4.1-mini, text only)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz
import pandas as pd
from Levenshtein import distance as lev_distance

from ..config import get_settings
from ..models import EntityRecord, EntityType, PipelineTraceRecord
from ..pipeline import PIPELINE_MODES, RaVIDPPipeline

ABLATION_MODES = list(PIPELINE_MODES)  # ["full", "gate_only", "no_rav"]


# ---------------------------------------------------------------------------
# ANLS
# ---------------------------------------------------------------------------

def _nls(pred: str, gt: str) -> float:
    pred, gt = pred.strip().lower(), gt.strip().lower()
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return lev_distance(pred, gt) / denom


def _anls_score(pred: str, gt_answers: list[str]) -> float:
    if not gt_answers:
        return 0.0
    return max(
        (1 - nls) if (nls := _nls(pred, gt)) < 0.5 else 0.0
        for gt in gt_answers
    )


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _build_context(entity_records: list[EntityRecord]) -> str:
    """Flat text context from extracted entities, ordered by page then y-position."""
    ordered = sorted(entity_records, key=lambda r: (r.page_index, r.bbox.y0))
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
            extracted = getattr(record.content, "extracted_text", None)
            desc = getattr(record.content, "description", None)
            if extracted:
                parts.append(extracted)
            elif desc:
                parts.append(f"[Figure: {desc}]")
    return "\n\n".join(parts)


def _build_context_for_mode(
    entity_records: list[EntityRecord],
    traces: list[PipelineTraceRecord],
    mode: str,
) -> str:
    """Build context for a given ablation mode from full-mode traces.

    Mode semantics:
      full      — use final_entity.content (fallback where triggered, else primary)
      no_rav    — use primary_entity.content for all entities (no filtering)
      gate_only — use primary_entity.content only for entities that passed the
                  fidelity gate; entities that failed are excluded from context
    """
    ordered = sorted(
        zip(entity_records, traces),
        key=lambda x: (x[0].page_index, x[0].bbox.y0),
    )
    parts: list[str] = []
    for record, trace in ordered:
        entity_type = record.entity_type

        if mode == "full":
            content = trace.final_entity.content
        elif mode == "no_rav":
            content = trace.primary_entity.content
        elif mode == "gate_only":
            # exclude entities that failed the fidelity gate
            pf = trace.primary_fidelity
            if pf is not None and not pf.passed_threshold:
                continue
            content = trace.primary_entity.content
        else:
            content = trace.final_entity.content

        if entity_type == EntityType.TEXT:
            text = getattr(content, "text", "").strip()
            if text:
                parts.append(text)
        elif entity_type == EntityType.TABLE:
            md = getattr(content, "markdown", "").strip()
            if md:
                parts.append(md)
        elif entity_type == EntityType.IMAGE:
            # images: always use final_entity (enriched); primary may lack description
            img_content = trace.final_entity.content
            extracted = getattr(img_content, "extracted_text", None)
            desc = getattr(img_content, "description", None)
            if extracted:
                parts.append(extracted)
            elif desc:
                parts.append(f"[Figure: {desc}]")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# QA via OpenAI (text-only — uses openai_qa_model)
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are a precise document question answering assistant. "
    "Answer the question using only information from the provided document context. "
    "Give a short, direct answer. If the answer is not in the context, reply with 'unanswerable'."
)


def _answer_question(context: str, question: str, settings) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_qa_model,
        max_tokens=128,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Image → single-page PDF conversion
# ---------------------------------------------------------------------------

def _image_bytes_to_pdf(image_bytes: bytes) -> bytes:
    """Wrap a raster image in a single-page PDF so the pipeline can process it.

    Page dimensions are fixed to standard A4 (595×842 pt) regardless of source
    image resolution, preventing enormous raster sizes during Docling rendering.
    """
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None  # suppress DecompressionBombWarning for large scans
    pil_img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # A4 portrait in points (1 pt = 1/72 inch); landscape if image is wider than tall
    w_px, h_px = pil_img.size
    if w_px > h_px:
        page_w, page_h = 842, 595  # A4 landscape
    else:
        page_w, page_h = 595, 842  # A4 portrait
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    page.insert_image(page.rect, stream=img_buf.getvalue())
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class QARecord:
    question_id: str
    doc_id: str
    question: str
    predicted_answer: str
    gt_answers: list[str]
    anls: float
    mode: str
    pipeline_error: str | None = None


@dataclass
class ModeSummary:
    mode: str
    num_questions: int
    num_docs: int
    mean_anls: float
    answerable_rate: float   # fraction of questions where anls > 0
    pipeline_error_rate: float


@dataclass
class Stage6Summary:
    split: str
    limit: int | None
    num_unique_docs: int
    modes: list[ModeSummary] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Checkpoint helpers — survive kills/restarts
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path) -> dict:
    """Load checkpoint dict: {doc_id: {mode: context_str} | {"error": str}}."""
    if ckpt_path.exists():
        try:
            return json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_checkpoint(ckpt_path: Path, checkpoint: dict) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(checkpoint), encoding="utf-8")


def _load_qa_checkpoint(ckpt_path: Path) -> list[dict]:
    """Load completed QA records from checkpoint."""
    if ckpt_path.exists():
        try:
            return json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_qa_checkpoint(ckpt_path: Path, records: list[dict]) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core benchmark — process docs once, derive all mode contexts from traces
# ---------------------------------------------------------------------------

def _run_ablation(
    modes: list[str],
    frame: pd.DataFrame,
    settings,
    checkpoint_dir: Path | None = None,
) -> tuple[int, list[QARecord]]:
    """Process each document once, then answer questions for every requested mode.

    Checkpoints doc contexts and QA records to disk after each step so the run
    can resume from where it left off if killed.

    Returns (num_unique_docs, qa_records).
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path("artifacts/stage6_checkpoints")

    doc_ckpt_path = checkpoint_dir / "doc_contexts.json"
    qa_ckpt_path = checkpoint_dir / "qa_records.json"

    # load existing progress
    doc_contexts: dict[str, dict | str] = _load_checkpoint(doc_ckpt_path)
    completed_qa: list[dict] = _load_qa_checkpoint(qa_ckpt_path)
    completed_keys = {(r["question_id"], r["mode"]) for r in completed_qa}

    pipeline = RaVIDPPipeline(mode="full")
    rows = list(frame.itertuples(index=False))

    # first pass: process each unique document once (skip already checkpointed)
    for i, row in enumerate(rows):
        doc_id = str(getattr(row, "docId", getattr(row, "doc_id", "")))
        if doc_id in doc_contexts:
            continue

        print(f"[doc {i+1}] processing doc_id={doc_id}", flush=True)
        image_col = row.image
        image_bytes = image_col["bytes"] if isinstance(image_col, dict) else image_col
        tmp_path = None
        try:
            pdf_bytes = _image_bytes_to_pdf(image_bytes)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            entity_records, traces = pipeline.run_with_traces(tmp_path)
            # store per-mode context strings (no binary data — checkpoint-safe)
            doc_contexts[doc_id] = {
                m: _build_context_for_mode(entity_records, traces, m)
                for m in ABLATION_MODES
            }
        except Exception as exc:
            doc_contexts[doc_id] = {"error": str(exc)}
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        _save_checkpoint(doc_ckpt_path, doc_contexts)
        print(f"[doc {i+1}] done — checkpoint saved ({len(doc_contexts)} docs total)", flush=True)

    # second pass: for each mode, answer questions (skip already completed)
    qa_records_dicts: list[dict] = list(completed_qa)
    for mode in modes:
        print(f"[qa] starting mode={mode}", flush=True)
        for j, row in enumerate(rows):
            question_id = str(getattr(row, "questionId", getattr(row, "question_id", "")))
            if (question_id, mode) in completed_keys:
                continue

            doc_id = str(getattr(row, "docId", getattr(row, "doc_id", question_id)))
            question = str(row.question)
            gt_answers = list(row.answers) if hasattr(row, "answers") else []

            cached = doc_contexts.get(doc_id, {})
            if isinstance(cached, dict) and "error" in cached:
                rec = dict(question_id=question_id, doc_id=doc_id, question=question,
                           predicted_answer="", gt_answers=gt_answers, anls=0.0,
                           mode=mode, pipeline_error=cached["error"])
            else:
                context = cached.get(mode, "") if isinstance(cached, dict) else ""
                if not context:
                    rec = dict(question_id=question_id, doc_id=doc_id, question=question,
                               predicted_answer="", gt_answers=gt_answers, anls=0.0,
                               mode=mode, pipeline_error="empty context")
                else:
                    try:
                        predicted = _answer_question(context, question, settings)
                        error = None
                    except Exception as exc:
                        predicted = ""
                        error = str(exc)
                    rec = dict(question_id=question_id, doc_id=doc_id, question=question,
                               predicted_answer=predicted, gt_answers=gt_answers,
                               anls=_anls_score(predicted, gt_answers),
                               mode=mode, pipeline_error=error)

            qa_records_dicts.append(rec)
            completed_keys.add((question_id, mode))

            if (j + 1) % 50 == 0:
                _save_qa_checkpoint(qa_ckpt_path, qa_records_dicts)
                print(f"[qa] mode={mode} q={j+1} — checkpoint saved", flush=True)

        _save_qa_checkpoint(qa_ckpt_path, qa_records_dicts)
        print(f"[qa] mode={mode} complete", flush=True)

    qa_records = [QARecord(**r) for r in qa_records_dicts]
    num_unique_docs = sum(
        1 for v in doc_contexts.values()
        if isinstance(v, dict) and "error" not in v
    )
    return num_unique_docs, qa_records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_endtoend_benchmark(
    dataset_root: str | Path,
    split: str = "val",
    limit: int | None = None,
    modes: list[str] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Stage6Summary, list[QARecord]]:
    """Run end-to-end DocVQA benchmark for one or more pipeline modes."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Stage 6.")

    if modes is None:
        modes = ["full"]
    for m in modes:
        if m not in PIPELINE_MODES:
            raise ValueError(f"Unknown mode {m!r}. Choose from {PIPELINE_MODES}.")

    dataset_root = Path(dataset_root)
    candidates = sorted(dataset_root.glob(f"data/{split}*.parquet"))
    if not candidates:
        candidates = sorted(dataset_root.glob(f"{split}*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files for split '{split}' under {dataset_root}. "
            "Download DocVQA: huggingface-cli download lmms-lab/DocVQA "
            "--repo-type dataset --local-dir data/raw/docvqa"
        )

    frame = pd.read_parquet(candidates[0])
    if limit is not None:
        frame = frame.head(limit)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path("artifacts/stage6_checkpoints")
    num_unique_docs, all_records = _run_ablation(modes, frame, settings, checkpoint_dir=ckpt_dir)

    mode_summaries: list[ModeSummary] = []
    for mode in modes:
        records = [r for r in all_records if r.mode == mode]
        n = len(records)
        if n == 0:
            continue
        mean_anls = sum(r.anls for r in records) / n
        answerable = sum(1 for r in records if r.anls > 0) / n
        error_rate = sum(1 for r in records if r.pipeline_error) / n
        num_docs = len({r.doc_id for r in records})

        mode_summaries.append(ModeSummary(
            mode=mode,
            num_questions=n,
            num_docs=num_docs,
            mean_anls=round(mean_anls, 4),
            answerable_rate=round(answerable, 4),
            pipeline_error_rate=round(error_rate, 4),
        ))

    summary = Stage6Summary(
        split=split,
        limit=limit,
        num_unique_docs=num_unique_docs,
        modes=mode_summaries,
    )
    return summary, all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 6 DocVQA end-to-end benchmark.")
    parser.add_argument("--dataset-root", default="data/raw/docvqa/DocVQA")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions (same rows used across all modes).")
    parser.add_argument(
        "--mode", default="full",
        choices=ABLATION_MODES + ["ablation"],
        help=(
            "Pipeline mode to run. "
            "'ablation' runs all three modes (no_rav, gate_only, full) "
            "on the same question set for direct comparison."
        ),
    )
    parser.add_argument("--output", default=None, help="JSON output file path.")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/stage6_checkpoints",
        help="Directory for resumable checkpoints (doc contexts + QA records).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    modes = ABLATION_MODES if args.mode == "ablation" else [args.mode]
    summary, records = run_endtoend_benchmark(
        dataset_root=args.dataset_root,
        split=args.split,
        limit=args.limit,
        modes=modes,
        checkpoint_dir=args.checkpoint_dir,
    )

    payload = {
        "summary": asdict(summary),
        "records": [asdict(r) for r in records],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
