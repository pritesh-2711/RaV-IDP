"""Stage 3c: text extraction benchmark on FUNSD."""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pytesseract
from Levenshtein import distance as lev_distance
from PIL import Image

from ..components.comparators.text import compare_text
from ..components.extractors.text import extract_text
from ..components.region_preprocessor import preprocess_region
from ..components.region_quality_classifier import classify_region
from ..components.reconstructors.text import reconstruct_text
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref_tokens)


def _ocr_extract(image_bytes: bytes, config: str = "--psm 11") -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return _normalize_text(pytesseract.image_to_string(image, config=config))


def _ground_truth_text(words: list[str] | tuple[str, ...]) -> str:
    return _normalize_text(" ".join(str(word) for word in words))


def _image_size(image_payload: dict) -> tuple[int, int]:
    image = Image.open(io.BytesIO(image_payload["bytes"]))
    return image.size


def _make_region(sample_id: str, image_payload: dict, extracted_text: str) -> DetectedRegion:
    width, height = _image_size(image_payload)
    return DetectedRegion(
        region_id=sample_id,
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=0, x1=width, y1=height, page=0),
        original_crop=image_payload["bytes"],
        processed_crop=image_payload["bytes"],
        raw_docling_record={"text": extracted_text},
        page_index=0,
    )


@dataclass
class TextBenchmarkRecord:
    sample_id: str
    ground_truth_text: str
    extracted_text: str
    reocr_text: str
    cer: float
    wer: float
    fidelity_score: float
    passed_threshold: bool


@dataclass
class TextBenchmarkSummary:
    split: str
    num_samples: int
    mean_cer: float
    median_cer: float
    mean_wer: float
    mean_fidelity: float
    pass_rate: float
    fidelity_cer_correlation: float


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def run_text_benchmark(
    dataset_root: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> tuple[TextBenchmarkSummary, list[TextBenchmarkRecord]]:
    """Run text extraction benchmark on FUNSD parquet data."""

    settings = get_settings()
    dataset_root = Path(dataset_root)
    parquet_path = dataset_root / "data" / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(parquet_path)
    if limit is not None:
        frame = frame.head(limit)

    records: list[TextBenchmarkRecord] = []
    for row in frame.itertuples(index=False):
        ground_truth = _ground_truth_text(list(row.words))
        region = preprocess_region(classify_region(_make_region(str(row.id), row.image, "")))
        extracted_text = _ocr_extract(region.processed_crop or region.original_crop, config="--psm 11")
        region = region.model_copy(update={"raw_docling_record": {"text": extracted_text}})
        entity = extract_text(region)
        reconstruction = reconstruct_text(
            entity,
            region,
            is_native_pdf=False,
            document_path="funsd-image.png",
        )
        fidelity = compare_text(
            reconstruction.content,
            entity.content.text,
            region.region_id,
            settings.threshold_text,
            entity_type=EntityType.TEXT,
        )

        cer = 0.0 if not ground_truth else lev_distance(extracted_text, ground_truth) / len(ground_truth)
        wer = _word_error_rate(ground_truth, extracted_text)
        records.append(
            TextBenchmarkRecord(
                sample_id=str(row.id),
                ground_truth_text=ground_truth,
                extracted_text=extracted_text,
                reocr_text=reconstruction.content.reocr_text,
                cer=cer,
                wer=wer,
                fidelity_score=fidelity.fidelity_score,
                passed_threshold=fidelity.passed_threshold,
            )
        )

    mean_cer = sum(record.cer for record in records) / len(records)
    mean_wer = sum(record.wer for record in records) / len(records)
    mean_fidelity = sum(record.fidelity_score for record in records) / len(records)
    median_cer = sorted(record.cer for record in records)[len(records) // 2]
    pass_rate = sum(1 for record in records if record.passed_threshold) / len(records)
    summary = TextBenchmarkSummary(
        split=split,
        num_samples=len(records),
        mean_cer=mean_cer,
        median_cer=median_cer,
        mean_wer=mean_wer,
        mean_fidelity=mean_fidelity,
        pass_rate=pass_rate,
        fidelity_cer_correlation=_pearson(
            [record.fidelity_score for record in records],
            [-record.cer for record in records],
        ),
    )
    return summary, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3c FUNSD text benchmark.")
    parser.add_argument(
        "--dataset-root",
        default="data/raw/funsd",
        help="Path to the FUNSD dataset root.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output file for summary and records.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_text_benchmark(args.dataset_root, split=args.split, limit=args.limit)
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
