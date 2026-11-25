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


BEGIN_MARKER = "<!-- BEGIN generated: aprobe report -->"
END_MARKER = "<!-- END generated: aprobe report -->"

NOT_RUN = """
> **No extraction run yet.** Every number and figure below is written here by
> `aprobe report --write`, which reads `results/`. Until
> [`notebooks/extract.ipynb`](notebooks/extract.ipynb) has been run on Colab and
> its output unzipped to `data/raw/run/`, there is nothing to report, and
> nothing is filled in by hand.
"""


def render_results_section(
    *,
    metrics,
    comparisons,
    controls,
    accuracy,
    selection,
    calibration,
    excluding_refusals,
    agreement: dict | None,
    examples: dict | None = None,
    errors: pd.DataFrame | None = None,
    headline_signal: str = "probe_before_linear",
) -> str:
    """The whole generated block, from the headline table to the controls."""
    from answer_probe.evaluate import best_baseline_column, signal_label

    headline = selection["selection"]["headline"]
    chosen = selection.get("best_confidence_signal")
    best_conf = (
        f"conf_{chosen}"
        if chosen and f"conf_{chosen}" in set(metrics["signal"])
        else best_baseline_column(metrics)
    )
    lines = [
        "### Headline",
        "",
        headline_table(metrics, headline_signal=headline_signal),
        "",
        f"The headline probe is a linear probe on the residual stream **before "
        f"answering**, at layer **{headline['layer']}**, with L2 "
        f"**{headline['l2']}**. Both were chosen on the validation split; the "
        f"test and shift columns were computed once, afterwards. Intervals are "
        f"95% percentile bootstrap over questions.",
        "",
        "![AUROC by layer](results/layer_curve.png)",
        "",
        "### Did the probe beat the model's own confidence?",
        "",
        f"The strongest confidence baseline on the validation split was "
        f"**{signal_label(best_conf)}**, so that is what the probe is measured "
        f"against here. The table above reports every baseline.",
        "",
        f"- {comparison_sentence(comparisons, split='test', other=best_conf)}",
        f"- {comparison_sentence(comparisons, split='shift', other=best_conf)}",
        "",
        "### Controls",
        "",
        f"- {controls_sentence(controls)}",
    ]

    surface = metrics[(metrics["split"] == "test") & (metrics["signal"] == "surface")]
    if not surface.empty:
        lines.append(
            f"- Question-surface baseline (question text only): AUROC "
            f"{surface.iloc[0]['auroc']:.3f} on the TriviaQA test split."
        )
    if agreement and agreement.get("n_graded"):
        lines.append(
            f"- Label audit: {agreement['agreement']:.1%} agreement with hand "
            f"grades on {agreement['n_graded']} answers "
            f"({agreement['n_disagree']} disagreements)."
        )

    lines += ["", "### Accuracy and calibration", "", f"{accuracy_sentence(accuracy)}", ""]
    if len(calibration):
        for _, row in calibration.iterrows():
            lines.append(
                f"- {row['split']}: expected calibration error {row['ece']:.3f}, "
                f"mean predicted P(wrong) {row['mean_predicted']:.3f} against an "
                f"observed {row['observed_error_rate']:.3f}."
            )
        lines += ["", "![Calibration](results/calibration.png)", ""]
    lines += ["![Selective answering](results/risk_coverage.png)", ""]

    if len(excluding_refusals):
        rows = excluding_refusals[
            (excluding_refusals["split"] == "test")
            & (excluding_refusals["signal"] == headline_signal)
        ]
        if not rows.empty:
            row = rows.iloc[0]
            lines += [
                "### With refusals excluded",
                "",
                f"The prompt lets the model reply \"unknown\", and a refusal is always "
                f"scored wrong, so part of what a probe detects could just be \"this "
                f"one is about to decline\". On the {int(row['n_committed'])} TriviaQA "
                f"test questions where the model committed to an answer, the probe's "
                f"AUROC is {row['auroc']:.3f} "
                f"[{row['auroc_low']:.3f}, {row['auroc_high']:.3f}].",
                "",
            ]

    if errors is not None and len(errors):
        lines += ["### Where the errors are", ""]
        main = errors[errors["dataset"] == "triviaqa"]
        by_type = main[main["grouping"] == "answer_type"].sort_values(
            "error_rate", ascending=False
        )
        if len(by_type):
            rows = ["| Answer type | Questions | Error rate | Refusal rate |",
                    "|---|---|---|---|"]
            for _, row in by_type.iterrows():
                rows.append(
                    f"| {row['value']} | {int(row['n'])} | {row['error_rate']:.3f} "
                    f"| {row['refusal_rate']:.3f} |"
                )
            lines += ["By the answer type guessed from the question word, on TriviaQA:",
                      "", "\n".join(rows), ""]
        refusal = main[main["grouping"] == "refusal"]
        if len(refusal) == 2:
            declined = refusal[refusal["value"] == "refused"].iloc[0]
            committed = refusal[refusal["value"] != "refused"].iloc[0]
            lines += [
                f"The model declined to answer {int(declined['n'])} TriviaQA questions "
                f"and committed to {int(committed['n'])}. A refusal is always scored "
                f"wrong; among the answers it did commit to, {committed['error_rate']:.1%} "
                "were wrong.",
                "",
            ]
        lines += ["Full breakdown, including by question length: "
                  "[`results/error_concentration.csv`](results/error_concentration.csv).",
                  ""]

    if examples:
        lines += [
            "### Where the probe and the confidence signal disagree",
            "",
            "Wrong answers on the TriviaQA test split, ranked by how far apart the "
            "two signals put them. Both columns are percentiles within the split, "
            "where 1.00 means \"most likely to be wrong\".",
            "",
        ]
        titles = {
            "probe_caught": "The probe flagged these; the confidence signal did not",
            "confidence_caught": "The confidence signal flagged these; the probe did not",
        }
        for key, title in titles.items():
            frame = examples.get(key)
            if frame is None or not len(frame):
                continue
            lines += [f"**{title}**", "", _examples_table(frame), ""]
    return "\n".join(lines).rstrip() + "\n"


def _examples_table(frame) -> str:
    header = ["Question", "Model answer", "Gold", "Probe pct", "Confidence pct"]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    _clip(row["question"], 70),
                    _clip(str(row["answer"]), 32),
                    _clip(str(row.get("gold_answer", "")), 28),
                    f"{row['probe_pct']:.2f}",
                    f"{row['conf_pct']:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def splice_into_readme(path, section: str) -> bool:
    """Replace the generated block in the README. Returns True if it changed."""
    from pathlib import Path

    path = Path(path)
    text = path.read_text()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise ValueError(
            f"{path} has no generated block; expected {BEGIN_MARKER} ... {END_MARKER}"
        )
    head, rest = text.split(BEGIN_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    updated = f"{head}{BEGIN_MARKER}\n\n{section}\n{END_MARKER}{tail}"
    if updated == text:
        return False
    path.write_text(updated)
    return True
