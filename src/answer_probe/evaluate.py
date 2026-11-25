"""Metrics, comparisons and plots, computed from the saved per-question scores.

This module never fits anything. It reads `results/scores.csv`, which `aprobe
train` wrote, so every reported number can be recomputed -- and checked -- from
a file in the repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from answer_probe.features import CONFIDENCE_SIGNALS
from answer_probe.metrics import (
    accuracy_at_coverage,
    auprc,
    auroc,
    baserate,
    bootstrap,
    bootstrap_difference,
    expected_calibration_error,
    reliability,
    risk_coverage,
)
from answer_probe.splits import SHIFT, TEST

REPORT_SPLITS = (TEST, SHIFT)
COVERAGES = (0.8, 0.5)
SPLIT_LABELS = {TEST: "TriviaQA test", SHIFT: "NQ-open (shift)"}

PROBE_LABELS = {
    "probe_before_linear": "Linear probe, before answering",
    "probe_after_linear": "Linear probe, after answering",
    "probe_before_diffmeans": "Difference-of-means, before answering",
    "probe_after_diffmeans": "Difference-of-means, after answering",
    "surface": "Question-surface baseline",
    "combination": "Probe + best confidence signal",
}


def signal_columns(scores: pd.DataFrame) -> list[str]:
    """Every scored signal in the frame, probes and baselines alike."""
    columns = [c for c in scores.columns if c.startswith(("probe_", "conf_"))]
    columns += [c for c in ("surface", "combination") if c in scores.columns]
    return columns


def signal_label(column: str) -> str:
    if column in PROBE_LABELS:
        return PROBE_LABELS[column]
    if column.startswith("conf_"):
        return CONFIDENCE_SIGNALS[column[len("conf_") :]]["label"]
    return column


def metrics_table(scores: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    """AUROC, AUPRC and selective-answering accuracy for every signal."""
    rows = []
    for split in REPORT_SPLITS:
        frame = scores[scores["split"] == split]
        if frame.empty:
            continue
        y = frame["wrong"].to_numpy(dtype=int)
        for column in signal_columns(frame):
            s = frame[column].to_numpy(dtype=float)
            roc = bootstrap(auroc, s, y, n_boot=n_boot)
            prc = bootstrap(auprc, s, y, n_boot=n_boot)
            row = {
                "split": split,
                "signal": column,
                "label": signal_label(column),
                "n": len(frame),
                "error_rate": round(baserate(y), 4),
                **roc.as_dict("auroc"),
                **prc.as_dict("auprc"),
            }
            for coverage in COVERAGES:
                row[f"acc_at_{int(coverage * 100)}"] = accuracy_at_coverage(s, y, coverage)
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(
    scores: pd.DataFrame, reference: str, *, n_boot: int
) -> pd.DataFrame:
    """Paired bootstrap of AUROC(reference) - AUROC(other) for every other signal.

    Paired on the same resampled questions, because the signals are computed on
    the same questions and share most of their variation.
    """
    rows = []
    for split in REPORT_SPLITS:
        frame = scores[scores["split"] == split]
        if frame.empty or reference not in frame:
            continue
        y = frame["wrong"].to_numpy(dtype=int)
        ref = frame[reference].to_numpy(dtype=float)
        for column in signal_columns(frame):
            if column == reference:
                continue
            other = frame[column].to_numpy(dtype=float)
            diff = bootstrap_difference(auroc, ref, other, y, n_boot=n_boot)
            rows.append(
                {
                    "split": split,
                    "reference": reference,
                    "other": column,
                    "other_label": signal_label(column),
                    **diff.as_dict("auroc_difference"),
                    "reference_better": bool(diff.low > 0),
                    "indistinguishable": bool(diff.low <= 0 <= diff.high),
                }
            )
    return pd.DataFrame(rows)


def risk_coverage_table(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Long-format risk-coverage curves, thinned to a readable grid."""
    rows = []
    for split in REPORT_SPLITS:
        frame = scores[scores["split"] == split]
        if frame.empty:
            continue
        y = frame["wrong"].to_numpy(dtype=int)
        for column in columns:
            if column not in frame:
                continue
            coverage, risk = risk_coverage(frame[column].to_numpy(dtype=float), y)
            # One point per percent of coverage is plenty for a plot and keeps
            # the committed file small.
            keep = np.unique(np.round(coverage * 100).astype(int), return_index=True)[1]
            for i in keep:
                rows.append(
                    {
                        "split": split,
                        "signal": column,
                        "label": signal_label(column),
                        "coverage": round(float(coverage[i]), 4),
                        "risk": round(float(risk[i]), 6),
                        "accuracy": round(float(1 - risk[i]), 6),
                    }
                )
    return pd.DataFrame(rows)


def calibration_table(scores: pd.DataFrame, prob_column: str, *, n_bins: int = 10):
    """Reliability bins and expected calibration error for a probability column."""
    rows = []
    summary = []
    for split in REPORT_SPLITS:
        frame = scores[scores["split"] == split]
        if frame.empty or prob_column not in frame:
            continue
        probs = frame[prob_column].to_numpy(dtype=float)
        y = frame["wrong"].to_numpy(dtype=int)
        rel = reliability(probs, y, n_bins=n_bins)
        for i in range(n_bins):
            rows.append(
                {
                    "split": split,
                    "bin_centre": float(rel.bin_centre[i]),
                    "mean_predicted": float(rel.mean_predicted[i]),
                    "observed": float(rel.observed[i]),
                    "count": int(rel.count[i]),
                }
            )
        summary.append(
            {
                "split": split,
                "n": len(frame),
                "ece": round(expected_calibration_error(probs, y, n_bins=n_bins), 4),
                "mean_predicted": round(float(probs.mean()), 4),
                "observed_error_rate": round(float(y.mean()), 4),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def best_baseline_column(metrics: pd.DataFrame, *, split: str = TEST) -> str:
    """The confidence baseline with the highest AUROC on a split.

    Used only to decide which line to draw on the plots and which row to put in
    the headline table; the probe-versus-baseline comparison reports every
    baseline, so this choice cannot flatter the probe.
    """
    frame = metrics[(metrics["split"] == split) & metrics["signal"].str.startswith("conf_")]
    if frame.empty:
        raise ValueError("no confidence baselines in the metrics table")
    return str(frame.loc[frame["auroc"].idxmax(), "signal"])


def refusal_breakdown(scores: pd.DataFrame, columns: list[str], *, n_boot: int) -> pd.DataFrame:
    """AUROC with refusals excluded.

    The prompt lets the model reply "unknown", and a refusal is always scored
    wrong. If a probe mostly detects "the model is about to decline", that is
    worth knowing, so the headline signals are also scored on the questions
    where the model committed to an answer.
    """
    rows = []
    for split in REPORT_SPLITS:
        frame = scores[(scores["split"] == split) & (scores["refused"] == 0)]
        if frame.empty:
            continue
        y = frame["wrong"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        for column in columns:
            if column not in frame:
                continue
            roc = bootstrap(auroc, frame[column].to_numpy(dtype=float), y, n_boot=n_boot)
            rows.append(
                {
                    "split": split,
                    "signal": column,
                    "label": signal_label(column),
                    "n_committed": len(frame),
                    "error_rate": round(baserate(y), 4),
                    **roc.as_dict("auroc"),
                }
            )
    return pd.DataFrame(rows)
