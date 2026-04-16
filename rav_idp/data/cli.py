"""CLI entrypoints for dataset acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .downloader import DatasetDownloader
from .registry import list_datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage RaV-IDP datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List datasets from the paper registry.")
    list_parser.add_argument("--json", action="store_true", help="Print dataset metadata as JSON.")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch or prepare datasets.")
    fetch_parser.add_argument("datasets", nargs="*", help="Dataset keys to fetch. Defaults to all.")

    stage_parser = subparsers.add_parser("stage", help="Stage an already downloaded dataset into the data root.")
    stage_parser.add_argument("dataset", help="Dataset key.")
    stage_parser.add_argument("path", help="Local source path for the dataset.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    downloader = DatasetDownloader()

    if args.command == "list":
        datasets = [dataset.model_dump() for dataset in list_datasets()]
        if args.json:
            print(json.dumps(datasets, indent=2))
        else:
            for dataset in datasets:
                print(f"{dataset['key']}: {dataset['display_name']} [{dataset['access']}] -> {dataset['stage']}")
        return 0

    if args.command == "fetch":
        results = downloader.fetch_many(args.datasets or None)
        for result in results:
            print(f"{result.dataset_key}: {result.status} - {result.message}")
        return 0

    if args.command == "stage":
        result = downloader.stage_external(args.dataset, Path(args.path))
        print(f"{result.dataset_key}: {result.status} - {result.message}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
