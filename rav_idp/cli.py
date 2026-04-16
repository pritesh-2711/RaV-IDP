"""CLI for running the RaV-IDP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import default_document_output, write_entity_records
from .pipeline import RaVIDPPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RaV-IDP pipeline on a document.")
    parser.add_argument("document", help="Path to the input document.")
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    pipeline = RaVIDPPipeline()
    records = pipeline.run(Path(args.document))
    output_path = Path(args.output) if args.output else default_document_output(args.document)
    written = write_entity_records(records, output_path)
    print(f"Wrote {len(records)} entity records to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
