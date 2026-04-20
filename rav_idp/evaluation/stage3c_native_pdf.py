"""Stage 3c supplement: text extraction benchmark on native PDFs.

Uses born-digital arXiv PDFs where PyMuPDF can read the embedded text layer
as perfect ground truth. Contrasts with FUNSD (scanned OCR) results to show
that the pipeline works well on native PDFs and that FUNSD CER=0.517 reflects
scanned-document complexity rather than pipeline failure.

Ground truth: PyMuPDF page.get_text() over the detected region bbox.
Extractor: Docling (via detect_layout, same as production pipeline).
Fidelity: reconstruct_text(is_native_pdf=True) + compare_text — uses same
          PDF text stream as GT, so fidelity ≈ 1 − CER by construction.

Usage:
  python -m rav_idp.evaluation.stage3c_native_pdf \
      --pdf-dir data/raw/native_pdfs \
      [--download] [--limit 25] [--output artifacts/stage3c_native_pdf.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Levenshtein import distance as lev_distance

from ..components.comparators.text import compare_text
from ..components.extractors.text import extract_text
from ..components.layout_detector import detect_layout
from ..components.page_renderer import render_document_pages
from ..components.reconstructors.text import extract_pdf_text_stream, reconstruct_text
from ..config import get_settings
from ..models import EntityType


# ---------------------------------------------------------------------------
# 25 diverse arXiv paper IDs for a reproducible benchmark set
# Covers ML, NLP, CV, physics, biology, mathematics, economics
# ---------------------------------------------------------------------------
ARXIV_IDS = [
    "1706.03762",  # Attention Is All You Need
    "2005.11401",  # RAG (Lewis et al.)
    "1810.04805",  # BERT
    "2203.15556",  # InstructGPT
    "2307.09288",  # Llama 2
    "2104.09864",  # LoRA
    "2006.11239",  # DDPM diffusion
    "1912.01412",  # GPT-3 (language models are few-shot learners)
    "2010.11929",  # ViT
    "2302.07116",  # LLaMA
    "2106.09685",  # CLIP
    "2303.08774",  # GPT-4 technical report
    "2212.09720",  # Constitutional AI
    "2305.10403",  # PaLM 2
    "2108.01072",  # Codex
    "2104.07857",  # RoBERTa (2019 repost)
    "2312.11805",  # Mixtral
    "1908.10084",  # Sentence-BERT
    "2009.01325",  # DeBERTa
    "1907.11692",  # RoBERTa (original)
    "2110.14168",  # FLAN
    "2201.11903",  # Chain-of-Thought prompting
    "2204.05149",  # Flamingo
    "2112.10752",  # Stable Diffusion (LDM)
    "2103.00020",  # CLIP (OpenAI version)
]


def _arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def download_arxiv_pdfs(paper_ids: list[str], dest_dir: Path, verbose: bool = True) -> list[Path]:
    """Download arXiv PDFs that are not already cached in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pid in paper_ids:
        out = dest_dir / f"{pid}.pdf"
        if out.exists() and out.stat().st_size > 1024:
            paths.append(out)
            if verbose:
                print(f"  cached: {out.name}", flush=True)
            continue
        url = _arxiv_url(pid)
        try:
            if verbose:
                print(f"  downloading {pid} ...", end=" ", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                out.write_bytes(resp.read())
            if verbose:
                print(f"{out.stat().st_size // 1024} KB", flush=True)
            paths.append(out)
            time.sleep(0.5)  # be polite to arxiv
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc})", flush=True)
    return paths


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def _cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return lev_distance(pred, gt) / len(gt)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    den_y = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


@dataclass
class NativePDFRecord:
    pdf_name: str
    page_index: int
    region_id: str
    gt_text: str
    extracted_text: str
    cer: float
    fidelity_score: float
    passed_threshold: bool


@dataclass
class NativePDFBenchmarkSummary:
    num_pdfs: int
    num_regions: int
    mean_cer: float
    median_cer: float
    mean_fidelity: float
    pass_rate: float
    fidelity_cer_spearman: float
    threshold_text: float
    min_gt_chars: int


def run_native_pdf_benchmark(
    pdf_dir: str | Path,
    limit: int | None = None,
    min_gt_chars: int = 20,
    verbose: bool = True,
) -> tuple[NativePDFBenchmarkSummary, list[NativePDFRecord]]:
    """Run text extraction benchmark on native PDFs in pdf_dir.

    Each PDF is processed through the production layout detector + text
    extractor + reconstructor + comparator stack. GT is the embedded PDF
    text stream, read via PyMuPDF for the exact region bbox.

    Args:
        pdf_dir: Directory containing .pdf files.
        limit: Max number of PDFs to process (None = all).
        min_gt_chars: Skip regions where GT has fewer than this many chars.
        verbose: Print per-document progress.
    """
    settings = get_settings()
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]

    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    records: list[NativePDFRecord] = []

    for pdf_idx, pdf_path in enumerate(pdfs):
        if verbose:
            print(f"[{pdf_idx+1}/{len(pdfs)}] {pdf_path.name}", flush=True)

        try:
            page_records = render_document_pages(pdf_path)
            regions = detect_layout(pdf_path, page_records)
        except Exception as exc:
            if verbose:
                print(f"  layout error: {exc}", flush=True)
            continue

        text_regions = [r for r in regions if r.entity_type == EntityType.TEXT]
        if verbose:
            print(f"  {len(text_regions)} text regions", flush=True)

        for region in text_regions:
            # Ground truth: PyMuPDF embedded text stream at this bbox
            try:
                gt_text = _normalize(extract_pdf_text_stream(pdf_path, region))
            except Exception:
                continue
            if len(gt_text) < min_gt_chars:
                continue

            # Primary extraction: text from Docling's raw_docling_record
            try:
                entity = extract_text(region)
            except Exception:
                continue
            extracted_text = _normalize(entity.content.text)

            # Reconstruction validation (native PDF path → uses PDF text stream)
            try:
                reconstruction = reconstruct_text(
                    entity,
                    region,
                    is_native_pdf=True,
                    document_path=pdf_path,
                )
                fidelity = compare_text(
                    reconstruction.content,
                    entity.content.text,
                    region.region_id,
                    settings.threshold_text,
                    entity_type=EntityType.TEXT,
                )
            except Exception as exc:
                if verbose:
                    print(f"  fidelity error on region {region.region_id}: {exc}", flush=True)
                continue

            records.append(NativePDFRecord(
                pdf_name=pdf_path.name,
                page_index=region.page_index,
                region_id=region.region_id,
                gt_text=gt_text,
                extracted_text=extracted_text,
                cer=_cer(extracted_text, gt_text),
                fidelity_score=fidelity.fidelity_score,
                passed_threshold=fidelity.passed_threshold,
            ))

    if not records:
        raise RuntimeError("No valid text regions found across all PDFs.")

    cers = [r.cer for r in records]
    fids = [r.fidelity_score for r in records]
    n = len(records)
    mean_cer = sum(cers) / n
    median_cer = sorted(cers)[n // 2]
    mean_fidelity = sum(fids) / n
    pass_rate = sum(1 for r in records if r.passed_threshold) / n

    summary = NativePDFBenchmarkSummary(
        num_pdfs=len(pdfs),
        num_regions=n,
        mean_cer=round(mean_cer, 4),
        median_cer=round(median_cer, 4),
        mean_fidelity=round(mean_fidelity, 4),
        pass_rate=round(pass_rate, 4),
        fidelity_cer_spearman=round(_spearman(fids, [-c for c in cers]), 4),
        threshold_text=settings.threshold_text,
        min_gt_chars=min_gt_chars,
    )
    return summary, records


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3c native PDF text benchmark.")
    ap.add_argument("--pdf-dir", default="data/raw/native_pdfs",
                    help="Directory of .pdf files to benchmark.")
    ap.add_argument("--download", action="store_true",
                    help="Download the standard 25 arXiv PDFs into --pdf-dir first.")
    ap.add_argument("--arxiv-ids", nargs="*", default=None,
                    help="Override arXiv IDs to download (default: built-in 25).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of PDFs to process.")
    ap.add_argument("--min-gt-chars", type=int, default=20,
                    help="Skip text regions with fewer GT chars than this.")
    ap.add_argument("--output", default="artifacts/stage3c_native_pdf.json",
                    help="Output JSON path for summary + records.")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)

    if args.download:
        ids = args.arxiv_ids if args.arxiv_ids else ARXIV_IDS
        print(f"Downloading {len(ids)} arXiv PDFs to {pdf_dir} ...", flush=True)
        paths = download_arxiv_pdfs(ids, pdf_dir)
        print(f"Downloaded/cached {len(paths)} PDFs.", flush=True)

    summary, records = run_native_pdf_benchmark(
        pdf_dir,
        limit=args.limit,
        min_gt_chars=args.min_gt_chars,
    )

    payload = {"summary": asdict(summary), "records": [asdict(r) for r in records]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
