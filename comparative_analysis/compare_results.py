"""Aggregate all baseline results and print a comparison table vs RaV-IDP."""

from __future__ import annotations

import json
from pathlib import Path


RAV_RESULTS = {
    "RaV full":      {"anls": 0.4224, "answerable": 0.4467, "errors": 0.0067},
    "RaV no_rav":    {"anls": 0.4206, "answerable": 0.4433, "errors": 0.0067},
    "RaV gate_only": {"anls": 0.1408, "answerable": 0.1433, "errors": 0.2967},
}

BASELINE_FILES = {
    "Docling alone":    "artifacts/baselines/docling_300_val.json",
    "Marker":           "artifacts/baselines/marker_300_val.json",
    "Unstructured":     "artifacts/baselines/unstructured_300_val.json",
    "LlamaParse":       "artifacts/baselines/llamaparse_300_val.json",
    "GPT-4.1 vision":   "artifacts/baselines/gpt4_vision_300_val.json",
}


def load_summary(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        s = d.get("summary", {})
        return {
            "anls": s.get("mean_anls"),
            "answerable": s.get("answerable_rate"),
            "errors": s.get("error_rate"),
        }
    except Exception:
        return None


def main():
    all_results = {}

    # baselines
    for name, path in BASELINE_FILES.items():
        s = load_summary(path)
        if s:
            all_results[name] = s
        else:
            all_results[name] = None  # not yet run

    # RaV reference
    all_results.update(RAV_RESULTS)

    # sort by ANLS descending (None last)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["anls"] if x[1] else -1,
        reverse=True,
    )

    print("\n" + "=" * 70)
    print(f"{'System':<22} {'ANLS':>8} {'Answr%':>8} {'Err%':>8}  Notes")
    print("-" * 70)
    for name, r in sorted_results:
        if r is None:
            print(f"{name:<22} {'NOT RUN':>8}")
            continue
        note = ""
        if name == "RaV full":
            note = "← our system"
        elif name == "RaV no_rav":
            note = "← RaV w/o fidelity gate"
        elif name == "RaV gate_only":
            note = "← gate, no fallback"
        anls_str = f"{r['anls']:.4f}" if r["anls"] is not None else "N/A"
        ans_str  = f"{r['answerable']*100:.1f}%" if r["answerable"] is not None else "N/A"
        err_str  = f"{r['errors']*100:.1f}%" if r["errors"] is not None else "N/A"
        print(f"{name:<22} {anls_str:>8} {ans_str:>8} {err_str:>8}  {note}")
    print("=" * 70)
    print(f"\nDataset: DocVQA val, 300 questions, 85 unique docs")
    print(f"QA model: gpt-4.1-mini (all except GPT-4.1 vision which uses gpt-4.1 directly)")


if __name__ == "__main__":
    main()
