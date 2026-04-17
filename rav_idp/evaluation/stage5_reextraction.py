"""Stage 5: fallback recovery rate benchmark.

Measures what fraction of failed stage 3a table extractions (fidelity < τ)
are recovered by the GPT-4o vision fallback. 'Recovered' means the fallback
extraction achieves fidelity >= τ on the same comparator used in stage 3a.

Inputs:
  - stage 3a artifact JSON  (produced by stage3a_tables.py --output)
  - PubTabNet archive        (same archive used in stage 3a)
  - OPENAI_API_KEY           (required; raises RuntimeError if absent)

Metrics reported:
  num_failed              records with passed_threshold=False in stage 3a
  recovery_rate           fraction where fallback fidelity >= threshold
  mean_fidelity_before    mean fidelity among failed records (primary extractor)
  mean_fidelity_after     mean fidelity among failed records (fallback extractor)
  delta_fidelity          mean_fidelity_after - mean_fidelity_before
  mean_fidelity_recovered mean fallback fidelity on recovered records only
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from ..components.comparators.table import compare_table
from ..components.fallback_extractor import call_vision_fallback
from ..components.reconstructors.table import reconstruct_table
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType


def _load_failed_records(artifact_path: Path, threshold: float) -> list[dict]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return [r for r in payload.get("records", []) if not r.get("passed_threshold", True)]


def _load_image_bytes(archive_path: Path, split: str, filenames: list[str]) -> dict[str, bytes]:
    remaining = set(filenames)
    found: dict[str, bytes] = {}
    candidates: dict[str, str] = {}
    for fn in filenames:
        candidates[f"pubtabnet/{split}/{fn}"] = fn
        candidates[f"pubtabnet/{fn}"] = fn

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


def _make_region(sample_id: str, image_bytes: bytes) -> DetectedRegion:
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


@dataclass
class RecoveryRecord:
    sample_id: str
    filename: str
    original_fidelity: float
    fallback_fidelity: float
    recovered: bool
    error: str | None


@dataclass
class RecoverySummary:
    split: str
    num_failed: int
    recovery_rate: float
    mean_fidelity_before: float
    mean_fidelity_after: float
    delta_fidelity: float
    mean_fidelity_recovered: float


def run_recovery_benchmark(
    artifact_path: str | Path,
    dataset_root: str | Path,
    split: str = "val",
) -> tuple[RecoverySummary, list[RecoveryRecord]]:
    """Run fallback recovery benchmark on failed stage 3a records."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for stage 5. "
            "Set it in .env or as an environment variable."
        )

    artifact_path = Path(artifact_path)
    dataset_root = Path(dataset_root)
    archive_path = dataset_root / "pubtabnet.tar.gz"

    failed = _load_failed_records(artifact_path, settings.threshold_table)
    if not failed:
        raise ValueError("No failed records found in stage 3a artifact. Nothing to recover.")

    filenames = [r["filename"] for r in failed]
    image_bytes_map = _load_image_bytes(archive_path, split, filenames)

    records: list[RecoveryRecord] = []
    for stage3a_rec in failed:
        filename = stage3a_rec["filename"]
        sample_id = stage3a_rec["sample_id"]
        original_fidelity = float(stage3a_rec["fidelity_score"])
        predicted_cols = stage3a_rec.get("predicted_cols")

        image_bytes = image_bytes_map.get(filename)
        if not image_bytes:
            records.append(RecoveryRecord(
                sample_id=sample_id,
                filename=filename,
                original_fidelity=original_fidelity,
                fallback_fidelity=0.0,
                recovered=False,
                error="image not found in archive",
            ))
            continue

        region = _make_region(sample_id, image_bytes)

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

        records.append(RecoveryRecord(
            sample_id=sample_id,
            filename=filename,
            original_fidelity=original_fidelity,
            fallback_fidelity=fallback_fidelity,
            recovered=fallback_fidelity >= settings.threshold_table,
            error=error,
        ))

    n = len(records)
    successful = [r for r in records if r.error is None]
    recovered = [r for r in records if r.recovered]

    mean_before = sum(r.original_fidelity for r in records) / n
    mean_after = sum(r.fallback_fidelity for r in records) / n
    mean_recovered = sum(r.fallback_fidelity for r in recovered) / len(recovered) if recovered else 0.0

    summary = RecoverySummary(
        split=split,
        num_failed=n,
        recovery_rate=len(recovered) / n,
        mean_fidelity_before=mean_before,
        mean_fidelity_after=mean_after,
        delta_fidelity=mean_after - mean_before,
        mean_fidelity_recovered=mean_recovered,
    )
    return summary, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 5 fallback recovery benchmark.")
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to stage 3a output JSON (e.g. artifacts/stage3a_val.json).",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/raw/pubtabnet",
        help="Path to PubTabNet dataset root (must contain pubtabnet.tar.gz).",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary, records = run_recovery_benchmark(
        artifact_path=args.artifact,
        dataset_root=args.dataset_root,
        split=args.split,
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
