"""The headline table, as Markdown, built from `results/metrics.csv`.

`aprobe report` prints exactly what the README shows, so the two cannot drift.
"""

from __future__ import annotations

import pandas as pd

from answer_probe.evaluate import SPLIT_LABELS
from answer_probe.splits import SHIFT, TEST


def _fmt_interval(row: pd.Series, prefix: str) -> str:
    return f"{row[prefix]:.3f} [{row[f'{prefix}_low']:.3f}, {row[f'{prefix}_high']:.3f}]"


def headline_table(
    metrics: pd.DataFrame, *, headline_signal: str, order: list[str] | None = None
) -> str:
    """Best probe against every confidence baseline, on both evaluation splits."""
    if order is None:
        order = [headline_signal] + [
            s
            for s in metrics["signal"].unique()
            if s.startswith("conf_") or s in ("surface", "combination")
        ]
    splits = [s for s in (TEST, SHIFT) if (metrics["split"] == s).any()]

    header = ["Signal"]
    for split in splits:
        header += [f"{SPLIT_LABELS[split]} AUROC", "acc @ 80% coverage"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for signal in order:
        cells = []
        label = signal
        for split in splits:
            row = metrics[(metrics["split"] == split) & (metrics["signal"] == signal)]
            if row.empty:
                cells += ["--", "--"]
                continue
            row = row.iloc[0]
            label = row["label"]
            cells += [_fmt_interval(row, "auroc"), f"{row['acc_at_80']:.3f}"]
        if signal == headline_signal:
            label = f"**{label}**"
            cells = [f"**{c}**" if c != "--" else c for c in cells]
        lines.append("| " + " | ".join([label, *cells]) + " |")

    # The accuracy floor: answering everything, with no abstention at all.
    floor = ["No abstention (answer everything)"]
    for split in splits:
        row = metrics[metrics["split"] == split]
        floor += ["--", f"{1 - row.iloc[0]['error_rate']:.3f}"] if not row.empty else ["--", "--"]
    lines.append("| " + " | ".join(floor) + " |")
    return "\n".join(lines)


def comparison_sentence(comparisons: pd.DataFrame, *, split: str, other: str) -> str:
    """One sentence on whether the probe beat a baseline, with the paired interval."""
    row = comparisons[(comparisons["split"] == split) & (comparisons["other"] == other)]
    if row.empty:
        return f"No comparison available against {other} on {split}."
    row = row.iloc[0]
    verdict = (
        "beats it"
        if row["reference_better"]
        else ("is indistinguishable from it" if row["indistinguishable"] else "loses to it")
    )
    return (
        f"On {SPLIT_LABELS.get(split, split)} the probe {verdict}: AUROC difference "
        f"{row['auroc_difference']:+.3f} "
        f"[{row['auroc_difference_low']:+.3f}, {row['auroc_difference_high']:+.3f}] "
        "(paired bootstrap over questions)."
    )


def controls_sentence(controls: pd.DataFrame) -> str:
    """One sentence on the shuffled-label control."""
    if controls.empty:
        return "The shuffled-label control was not run."
    held_out = controls["linear_auroc_test"]
    in_sample = controls["linear_auroc_train_insample"]
    return (
        f"Shuffled labels, {len(controls)} permutations: held-out AUROC "
        f"{held_out.mean():.3f} (range {held_out.min():.3f}-{held_out.max():.3f}), "
        f"against {in_sample.mean():.3f} in sample on the permuted training labels."
    )


def accuracy_sentence(accuracy: pd.DataFrame) -> str:
    rows = accuracy[accuracy["split"] == "all"]
    parts = [
        f"{row['dataset']}: {row['accuracy']:.1%} correct, {row['refusal_rate']:.1%} refused"
        for _, row in rows.iterrows()
    ]
    return "Model accuracy -- " + "; ".join(parts) + "."
