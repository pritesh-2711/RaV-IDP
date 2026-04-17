"""CLI for running the RaV-IDP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inspection import VisualArtifactRecorder
from .io import default_document_output, default_visual_run_dir, write_entity_records
from .pipeline import RaVIDPPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RaV-IDP pipeline on a document.")
    parser.add_argument("document", help="Path to the input document.")
    parser.add_argument("--output", help="Optional output JSON path.")
    parser.add_argument("--run-dir", help="Optional directory for visual step-by-step artifacts.")
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Disable visual artifact capture and only write the final entity JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    pipeline = RaVIDPPipeline()
    recorder = None
    if not args.no_visuals:
        run_dir = Path(args.run_dir) if args.run_dir else default_visual_run_dir(args.document)
        recorder = VisualArtifactRecorder(run_dir)
    records = pipeline.run(Path(args.document), artifact_recorder=recorder)
    if args.output:
        output_path = Path(args.output)
    elif recorder is not None:
        output_path = recorder.run_dir / "05_final_output" / "entity_records.json"
    else:
        output_path = default_document_output(args.document)
    written = write_entity_records(records, output_path)
    print(f"Wrote {len(records)} entity records to {written}")
    if recorder is not None:
        print(f"Visual artifacts stored in {recorder.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
