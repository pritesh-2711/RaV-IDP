"""Stage 4: fidelity reliability study from Stage 3 artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.stats import spearmanr


@dataclass
class CorrelationSummary:
    num_samples: int
    spearman_rho: float
    p_value: float
    quality_field: str
    fidelity_field: str = "fidelity_score"


@dataclass
class ThresholdSummary:
    gt_positive_threshold: float
    best_tau: float
    best_f1: float
    precision: float
    recall: float
    positives: int
    negatives: int


@dataclass
class Stage4Summary:
    table_vs_cer: CorrelationSummary | None
    table_vs_teds: CorrelationSummary | None
    image_vs_phash: CorrelationSummary | None
    text_vs_cer: CorrelationSummary | None
    table_threshold: ThresholdSummary | None
    image_threshold: ThresholdSummary | None
    text_threshold: ThresholdSummary | None


def _load_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 3 or len(ys) < 3:
        return 0.0, 1.0
    rho, pval = spearmanr(xs, ys)
    if rho != rho:  # NaN guard
        return 0.0, 1.0
    return float(rho), float(pval)


def _binary_f1(predicted: list[bool], actual: list[bool]) -> tuple[float, float, float]:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum((not p) and a for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def _best_threshold(
    fidelity_scores: list[float],
    gt_positive: list[bool],
    gt_positive_threshold: float,
) -> ThresholdSummary:
    best_tau = 0.0
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    for step in range(0, 101):
        tau = step / 100.0
        predicted = [score >= tau for score in fidelity_scores]
        f1, precision, recall = _binary_f1(predicted, gt_positive)
        if f1 > best_f1:
            best_tau = tau
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    positives = sum(gt_positive)
    return ThresholdSummary(
        gt_positive_threshold=gt_positive_threshold,
        best_tau=best_tau,
        best_f1=best_f1,
        precision=best_precision,
        recall=best_recall,
        positives=positives,
        negatives=len(gt_positive) - positives,
    )


def _table_correlations(
    records: list[dict],
    cer_accept_threshold: float,
) -> tuple[CorrelationSummary | None, CorrelationSummary | None, ThresholdSummary | None]:
    if not records:
        return None, None, None

    fidelity = [float(r["fidelity_score"]) for r in records]
    # Stage 3a uses "cell_text_cer" as the ground-truth quality signal
    gt_cer = [float(r["cell_text_cer"]) for r in records]
    teds = [float(r["teds"]) for r in records if "teds" in r]

    rho, pval = _safe_spearman(fidelity, [-v for v in gt_cer])
    corr_cer = CorrelationSummary(
        num_samples=len(records),
        spearman_rho=rho,
        p_value=pval,
        quality_field="negative_cell_text_cer",
    )

    corr_teds = None
    if len(teds) == len(records):
        rho_t, pval_t = _safe_spearman(fidelity, teds)
        corr_teds = CorrelationSummary(
            num_samples=len(records),
            spearman_rho=rho_t,
            p_value=pval_t,
            quality_field="teds",
        )

    threshold = _best_threshold(
        fidelity_scores=fidelity,
        gt_positive=[v <= cer_accept_threshold for v in gt_cer],
        gt_positive_threshold=cer_accept_threshold,
    )
    return corr_cer, corr_teds, threshold


def _image_correlations(
    records: list[dict],
    phash_accept_threshold: float,
) -> tuple[CorrelationSummary | None, ThresholdSummary | None]:
    if not records:
        return None, None

    fidelity = [float(r["fidelity_score"]) for r in records]
    phash = [float(r["phash_similarity"]) for r in records]

    rho, pval = _safe_spearman(fidelity, phash)
    corr = CorrelationSummary(
        num_samples=len(records),
        spearman_rho=rho,
        p_value=pval,
        quality_field="phash_similarity",
    )

    threshold = _best_threshold(
        fidelity_scores=fidelity,
        gt_positive=[v >= phash_accept_threshold for v in phash],
        gt_positive_threshold=phash_accept_threshold,
    )
    return corr, threshold


def _text_correlations(
    records: list[dict],
    cer_accept_threshold: float,
) -> tuple[CorrelationSummary | None, ThresholdSummary | None]:
    if not records:
        return None, None

    fidelity = [float(r["fidelity_score"]) for r in records]
    cer = [float(r["cer"]) for r in records]

    rho, pval = _safe_spearman(fidelity, [-v for v in cer])
    corr = CorrelationSummary(
        num_samples=len(records),
        spearman_rho=rho,
        p_value=pval,
        quality_field="negative_cer",
    )

    threshold = _best_threshold(
        fidelity_scores=fidelity,
        gt_positive=[v <= cer_accept_threshold for v in cer],
        gt_positive_threshold=cer_accept_threshold,
    )
    return corr, threshold


def run_stage4(
    table_artifact: str | Path | None = None,
    image_artifact: str | Path | None = None,
    text_artifact: str | Path | None = None,
    table_cer_accept_threshold: float = 0.5,
    image_phash_accept_threshold: float = 0.85,
    text_cer_accept_threshold: float = 0.2,
) -> Stage4Summary:
    table_vs_cer = table_vs_teds = table_threshold = None
    image_vs_phash = image_threshold = None
    text_vs_cer = text_threshold = None

    if table_artifact:
        payload = _load_payload(table_artifact)
        table_vs_cer, table_vs_teds, table_threshold = _table_correlations(
            payload.get("records", []),
            cer_accept_threshold=table_cer_accept_threshold,
        )

    if image_artifact:
        payload = _load_payload(image_artifact)
        image_vs_phash, image_threshold = _image_correlations(
            payload.get("records", []),
            phash_accept_threshold=image_phash_accept_threshold,
        )

    if text_artifact:
        payload = _load_payload(text_artifact)
        text_vs_cer, text_threshold = _text_correlations(
            payload.get("records", []),
            cer_accept_threshold=text_cer_accept_threshold,
        )

    return Stage4Summary(
        table_vs_cer=table_vs_cer,
        table_vs_teds=table_vs_teds,
        image_vs_phash=image_vs_phash,
        text_vs_cer=text_vs_cer,
        table_threshold=table_threshold,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 4 fidelity reliability study from Stage 3 artifact files."
    )
    parser.add_argument("--table-artifact", default=None, help="Stage 3a artifact JSON.")
    parser.add_argument("--image-artifact", default=None, help="Stage 3b artifact JSON.")
    parser.add_argument("--text-artifact", default=None, help="Stage 3c artifact JSON.")
    parser.add_argument("--table-cer-accept-threshold", type=float, default=0.5)
    parser.add_argument("--image-phash-accept-threshold", type=float, default=0.85)
    parser.add_argument("--text-cer-accept-threshold", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_stage4(
        table_artifact=args.table_artifact,
        image_artifact=args.image_artifact,
        text_artifact=args.text_artifact,
        table_cer_accept_threshold=args.table_cer_accept_threshold,
        image_phash_accept_threshold=args.image_phash_accept_threshold,
        text_cer_accept_threshold=args.text_cer_accept_threshold,
    )
    payload = asdict(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
