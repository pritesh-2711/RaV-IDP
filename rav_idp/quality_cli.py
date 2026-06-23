"""CLI for profiling, calibration benchmarks, and experimental evidence generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .components.docres_backend import DocResConfig, DocResSubprocessBackend, VALID_DOCRES_TASKS
from .components.docentr_backend import DocEnTRConfig, DocEnTRSubprocessBackend
from .components.evidence_chain import build_page_evidence
from .components.page_renderer import render_document_pages
from .components.quality_profiler import profile_pages
from .evaluation.quality_benchmark import run_quality_benchmark


def _input_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _profile_command(args: argparse.Namespace) -> int:
    document = Path(args.document).expanduser().resolve()
    pages = profile_pages(render_document_pages(document), document)
    payload = {
        "input_id": _input_id(document),
        "pages": [
            page.quality_assessment.model_dump(mode="json")
            for page in pages
            if page.quality_assessment is not None
        ],
        "privacy": "The source filename and path are intentionally omitted.",
    }
    _write_json(Path(args.output), payload)
    print(f"Profiled {len(pages)} page(s); report written to {Path(args.output).resolve()}")
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    summary = run_quality_benchmark(
        args.input_dir,
        args.output_dir,
        max_documents=args.max_documents,
        max_pages_per_document=args.max_pages,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _restore_command(args: argparse.Namespace) -> int:
    document = Path(args.document).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = profile_pages(render_document_pages(document), document)
    backend = None
    if args.backend == "docres":
        repo_value = args.docres_repo or os.getenv("RAV_DOCRES_REPO")
        python_value = args.docres_python or os.getenv("RAV_DOCRES_PYTHON")
        if not repo_value or not python_value:
            raise SystemExit(
                "DocRes requires --docres-repo and --docres-python "
                "(or RAV_DOCRES_REPO and RAV_DOCRES_PYTHON)."
            )
        model_value = args.docres_model or os.getenv("RAV_DOCRES_MODEL")
        backend = DocResSubprocessBackend(
            DocResConfig(
                repo_dir=Path(repo_value),
                python_bin=python_value,
                model_path=Path(model_value) if model_value else None,
                task=args.docres_task,
                mode=args.docres_mode,
                max_input_dimension=(
                    args.docres_max_dimension if args.docres_max_dimension > 0 else None
                ),
                timeout_seconds=args.docres_timeout,
            )
        )
    elif args.backend == "docentr":
        repo_value = args.docentr_repo or os.getenv("RAV_DOCENTR_REPO")
        python_value = args.docentr_python or os.getenv("RAV_DOCENTR_PYTHON")
        weights_value = args.docentr_weights or os.getenv("RAV_DOCENTR_WEIGHTS")
        if not repo_value or not python_value or not weights_value:
            raise SystemExit(
                "DocEnTR requires --docentr-repo, --docentr-python, and --docentr-weights "
                "(or their RAV_DOCENTR_* environment variables)."
            )
        backend = DocEnTRSubprocessBackend(
            DocEnTRConfig(
                repo_dir=Path(repo_value),
                python_bin=python_value,
                weights_path=Path(weights_value),
                model_size=args.docentr_model_size,
                split_size=args.docentr_split_size,
                patch_size=args.docentr_patch_size,
                threshold=args.docentr_threshold,
                batch_size=args.docentr_batch_size,
                device=args.docentr_device,
                timeout_seconds=args.docentr_timeout,
            )
        )
    evidence = build_page_evidence(
        pages,
        enable_restoration=args.enable_restoration,
        backend=backend,
    )
    manifest = {
        "input_id": _input_id(document),
        "restoration_enabled": args.enable_restoration,
        "requested_backend": args.backend,
        "pages": [],
        "privacy": "The source filename and path are intentionally omitted.",
    }
    for bundle in evidence:
        stem = f"page_{bundle.page_index:03d}"
        (output_dir / f"{stem}_source_v1.png").write_bytes(bundle.source_page_v1)
        if bundle.candidate_page_v2 is not None:
            (output_dir / f"{stem}_candidate_v2.png").write_bytes(bundle.candidate_page_v2)
        (output_dir / f"{stem}_working_v2.png").write_bytes(bundle.working_page_v2)
        manifest["pages"].append(
            bundle.model_dump(
                mode="json",
                exclude={"source_page_v1", "candidate_page_v2", "working_page_v2"},
            )
        )
    _write_json(output_dir / "evidence_manifest.json", manifest)
    mode = "experimental restoration" if args.enable_restoration else "identity v2"
    print(f"Built {len(evidence)} page evidence bundle(s) in {mode} mode at {output_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RaV-IDP quality and restoration experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Profile one document without restoration.")
    profile.add_argument("document")
    profile.add_argument("--output", required=True)
    profile.set_defaults(handler=_profile_command)

    benchmark = subparsers.add_parser("benchmark", help="Run controlled synthetic calibration.")
    benchmark.add_argument("input_dir")
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--max-documents", type=int)
    benchmark.add_argument("--max-pages", type=int, default=3)
    benchmark.set_defaults(handler=_benchmark_command)

    restore = subparsers.add_parser("restore", help="Build versioned v1/v2 page evidence.")
    restore.add_argument("document")
    restore.add_argument("--output-dir", required=True)
    restore.add_argument(
        "--backend",
        choices=("deterministic", "docres", "docentr"),
        default="deterministic",
    )
    restore.add_argument(
        "--enable-restoration",
        action="store_true",
        help="Execute experimental transforms; omission guarantees byte-identical v2 evidence.",
    )
    restore.add_argument("--docres-repo", help="Official DocRes repository directory.")
    restore.add_argument("--docres-python", help="Python executable from the isolated DocRes environment.")
    restore.add_argument("--docres-model", help="Optional docres.pkl path; defaults inside the repository.")
    restore.add_argument(
        "--docres-task",
        choices=("auto", *sorted(VALID_DOCRES_TASKS)),
        default="auto",
    )
    restore.add_argument("--docres-timeout", type=int, default=600)
    restore.add_argument(
        "--docres-mode",
        choices=("opencv_then_docres", "docres_only"),
        default="opencv_then_docres",
        help="Run DocRes after the OpenCV plan, or after geometry correction only.",
    )
    restore.add_argument(
        "--docres-max-dimension",
        type=int,
        default=1024,
        help="Bound DocRes input for GPU memory; use 0 to disable resizing.",
    )
    restore.add_argument("--docentr-repo", help="Official DocEnTR repository directory.")
    restore.add_argument("--docentr-python", help="Python executable from the DocEnTR environment.")
    restore.add_argument("--docentr-weights", help="DocEnTR pretrained weight file.")
    restore.add_argument(
        "--docentr-model-size",
        choices=("small", "base", "large"),
        default="base",
    )
    restore.add_argument("--docentr-split-size", type=int, default=256)
    restore.add_argument("--docentr-patch-size", type=int, default=8)
    restore.add_argument("--docentr-threshold", type=float, default=0.5)
    restore.add_argument("--docentr-batch-size", type=int, default=1)
    restore.add_argument("--docentr-device")
    restore.add_argument("--docentr-timeout", type=int, default=600)
    restore.set_defaults(handler=_restore_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
