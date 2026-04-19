"""Stage 5: fallback recovery rate benchmark.

Measures what fraction of failed Stage 3 extractions are recovered by the
GPT-4o vision fallback. "Recovered" means the fallback extraction achieves
the same fidelity threshold used in the corresponding Stage 3 benchmark.

Supported recovery studies:
  - Tables: failed Stage 3a records from PubTabNet
  - Text:   failed Stage 3c records from FUNSD
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from ..components.comparators.table import compare_table
from ..components.comparators.text import compare_text
from ..components.fallback_extractor import call_vision_fallback
from ..components.reconstructors.table import reconstruct_table
from ..components.reconstructors.text import reconstruct_text
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType


def _load_failed_records(artifact_path: Path) -> list[dict]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return [record for record in payload.get("records", []) if not record.get("passed_threshold", True)]


def _load_pubtabnet_image_bytes(archive_path: Path, split: str, filenames: list[str]) -> dict[str, bytes]:
    remaining = set(filenames)
    found: dict[str, bytes] = {}
    candidates: dict[str, str] = {}
    for filename in filenames:
        candidates[f"pubtabnet/{split}/{filename}"] = filename
        candidates[f"pubtabnet/{filename}"] = filename

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = candidates.get(member.name)
            if name is None:
                continue
            handle = archive.extractfile(member)
            if handle:
                found[name] = handle.read()
            remaining.discard(name)
            if not remaining:
                break
    return found


def _make_table_region(sample_id: str, image_bytes: bytes) -> DetectedRegion:
    width, height = Image.open(io.BytesIO(image_bytes)).size
    return DetectedRegion(
        region_id=sample_id,
        entity_type=EntityType.TABLE,
        bbox=BoundingBox(x0=0, y0=0, x1=width, y1=height, page=0),
        original_crop=image_bytes,
        processed_crop=image_bytes,
        raw_docling_record={},
        page_index=0,
    )


def _make_text_region(sample_id: str, image_bytes: bytes) -> DetectedRegion:
    width, height = Image.open(io.BytesIO(image_bytes)).size
    return DetectedRegion(
        region_id=sample_id,
        entity_type=EntityType.TEXT,
        bbox=BoundingBox(x0=0, y0=0, x1=width, y1=height, page=0),
        original_crop=image_bytes,
        processed_crop=image_bytes,
        raw_docling_record={},
        page_index=0,
    )


@dataclass
class RecoveryRecord:
    sample_id: str
    original_fidelity: float
    fallback_fidelity: float
    recovered: bool
    error: str | None


@dataclass
class RecoverySummary:
    num_failed: int
    recovery_rate: float
    mean_fidelity_before: float
    mean_fidelity_after: float
    delta_fidelity: float
    mean_fidelity_recovered: float


@dataclass
class Stage5Summary:
    table_recovery: RecoverySummary | None
    text_recovery: RecoverySummary | None


def _summarize(records: list[RecoveryRecord]) -> RecoverySummary:
    n = len(records)
    recovered = [record for record in records if record.recovered]
    mean_before = sum(record.original_fidelity for record in records) / n if n else 0.0
    mean_after = sum(record.fallback_fidelity for record in records) / n if n else 0.0
    mean_recovered = (
        sum(record.fallback_fidelity for record in recovered) / len(recovered)
        if recovered
        else 0.0
    )
    return RecoverySummary(
        num_failed=n,
        recovery_rate=len(recovered) / n if n else 0.0,
        mean_fidelity_before=mean_before,
        mean_fidelity_after=mean_after,
        delta_fidelity=mean_after - mean_before,
        mean_fidelity_recovered=mean_recovered,
    )


def run_table_recovery_benchmark(
    artifact_path: str | Path,
    dataset_root: str | Path,
    split: str = "val",
) -> tuple[RecoverySummary, list[RecoveryRecord]]:
    """Run fallback recovery benchmark on failed Stage 3a records."""

    settings = get_settings()
    artifact_path = Path(artifact_path)
    dataset_root = Path(dataset_root)
    archive_path = dataset_root / "pubtabnet.tar.gz"

    failed = _load_failed_records(artifact_path)
    if not failed:
        raise ValueError("No failed records found in stage 3a artifact. Nothing to recover.")

    filenames = [record["filename"] for record in failed]
    image_bytes_map = _load_pubtabnet_image_bytes(archive_path, split, filenames)

    records: list[RecoveryRecord] = []
    for stage3_record in failed:
        filename = stage3_record["filename"]
        sample_id = stage3_record["sample_id"]
        original_fidelity = float(stage3_record["fidelity_score"])
        predicted_cols = stage3_record.get("predicted_cols")

        image_bytes = image_bytes_map.get(filename)
        if not image_bytes:
            records.append(
                RecoveryRecord(
                    sample_id=sample_id,
                    original_fidelity=original_fidelity,
                    fallback_fidelity=0.0,
                    recovered=False,
                    error="image not found in archive",
                )
            )
            continue

        region = _make_table_region(sample_id, image_bytes)

        try:
            fallback_entity = call_vision_fallback(region, context_text="")
            reconstruction = reconstruct_table(fallback_entity, region)
            fallback_fidelity_result = compare_table(
                reconstruction.content,
                region,
                settings.threshold_table,
                skip_visual=True,
                detected_col_count=predicted_cols,
            )
            fallback_fidelity = fallback_fidelity_result.fidelity_score
            error = None
        except Exception as exc:
            fallback_fidelity = 0.0
            error = str(exc)

        records.append(
            RecoveryRecord(
                sample_id=sample_id,
                original_fidelity=original_fidelity,
                fallback_fidelity=fallback_fidelity,
                recovered=fallback_fidelity >= settings.threshold_table,
                error=error,
            )
        )

    return _summarize(records), records


def _load_funsd_image_bytes(dataset_root: Path, split: str) -> dict[str, bytes]:
    parquet_path = dataset_root / "data" / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(parquet_path)
    image_map: dict[str, bytes] = {}
    for row in frame.itertuples(index=False):
        image_payload = row.image
        image_bytes = image_payload["bytes"] if isinstance(image_payload, dict) else image_payload
        image_map[str(row.id)] = image_bytes
    return image_map


def run_text_recovery_benchmark(
    artifact_path: str | Path,
    dataset_root: str | Path,
    split: str = "train",
) -> tuple[RecoverySummary, list[RecoveryRecord]]:
    """Run fallback recovery benchmark on failed Stage 3c records."""

    settings = get_settings()
    artifact_path = Path(artifact_path)
    dataset_root = Path(dataset_root)

    failed = _load_failed_records(artifact_path)
    if not failed:
        raise ValueError("No failed records found in stage 3c artifact. Nothing to recover.")

    image_bytes_map = _load_funsd_image_bytes(dataset_root, split)

    records: list[RecoveryRecord] = []
    for stage3_record in failed:
        sample_id = str(stage3_record["sample_id"])
        original_fidelity = float(stage3_record["fidelity_score"])
        image_bytes = image_bytes_map.get(sample_id)

        if not image_bytes:
            records.append(
                RecoveryRecord(
                    sample_id=sample_id,
                    original_fidelity=original_fidelity,
                    fallback_fidelity=0.0,
                    recovered=False,
                    error="image not found in dataset",
                )
            )
            continue

        region = _make_text_region(sample_id, image_bytes)

        try:
            fallback_entity = call_vision_fallback(region, context_text="")
            reconstruction = reconstruct_text(
                fallback_entity,
                region,
                is_native_pdf=False,
                document_path="funsd-image.png",
            )
            fallback_fidelity_result = compare_text(
                reconstruction.content,
                fallback_entity.content.text,
                region.region_id,
                settings.threshold_text,
                entity_type=EntityType.TEXT,
            )
            fallback_fidelity = fallback_fidelity_result.fidelity_score
            error = None
        except Exception as exc:
            fallback_fidelity = 0.0
            error = str(exc)

        records.append(
            RecoveryRecord(
                sample_id=sample_id,
                original_fidelity=original_fidelity,
                fallback_fidelity=fallback_fidelity,
                recovered=fallback_fidelity >= settings.threshold_text,
                error=error,
            )
        )

    return _summarize(records), records


def run_stage5(
    table_artifact: str | Path | None = None,
    table_dataset_root: str | Path = "data/raw/pubtabnet",
    table_split: str = "val",
    text_artifact: str | Path | None = None,
    text_dataset_root: str | Path = "data/raw/funsd",
    text_split: str = "train",
) -> tuple[Stage5Summary, dict[str, list[dict]]]:
    """Run Stage 5 recovery studies for any supplied Stage 3 artifacts."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for stage 5. "
            "Set it in .env or as an environment variable."
        )

    table_summary = None
    text_summary = None
    payload_records: dict[str, list[dict]] = {"table_records": [], "text_records": []}

    if table_artifact:
        table_summary, table_records = run_table_recovery_benchmark(
            artifact_path=table_artifact,
            dataset_root=table_dataset_root,
            split=table_split,
        )
        payload_records["table_records"] = [asdict(record) for record in table_records]

    if text_artifact:
        text_summary, text_records = run_text_recovery_benchmark(
            artifact_path=text_artifact,
            dataset_root=text_dataset_root,
            split=text_split,
        )
        payload_records["text_records"] = [asdict(record) for record in text_records]

    if table_summary is None and text_summary is None:
        raise ValueError("At least one of --table-artifact or --text-artifact must be provided.")

    return Stage5Summary(table_recovery=table_summary, text_recovery=text_summary), payload_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 5 fallback recovery benchmark.")
    parser.add_argument("--table-artifact", default=None, help="Path to Stage 3a output JSON.")
    parser.add_argument(
        "--table-dataset-root",
        default="data/raw/pubtabnet",
        help="Path to PubTabNet dataset root (must contain pubtabnet.tar.gz).",
    )
    parser.add_argument("--table-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--text-artifact", default=None, help="Path to Stage 3c output JSON.")
    parser.add_argument(
        "--text-dataset-root",
        default="data/raw/funsd",
        help="Path to FUNSD dataset root.",
    )
    parser.add_argument("--text-split", default="train", choices=["train", "test"])
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_stage5(
        table_artifact=args.table_artifact,
        table_dataset_root=args.table_dataset_root,
        table_split=args.table_split,
        text_artifact=args.text_artifact,
        text_dataset_root=args.text_dataset_root,
        text_split=args.text_split,
    )
    payload = {
        "summary": asdict(summary),
        **records,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
