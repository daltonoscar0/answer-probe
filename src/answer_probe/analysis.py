"""What the probe found: where errors sit, and where the probe and the model's
own confidence disagree.

The disagreement examples are picked by rank, not by raw score, because the
probe's logit and a probability are not on the same scale. For each question
both signals are turned into a percentile within its split, and the examples are
the wrong answers where the two percentiles are furthest apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUESTION_LENGTH_BINS = (0, 8, 12, 16, 20, 1000)
QUESTION_LENGTH_LABELS = ("<=8", "9-12", "13-16", "17-20", ">20")


def _rate_row(name: str, value: str, group: pd.DataFrame) -> dict:
    return {
        "grouping": name,
        "value": value,
        "n": len(group),
        "error_rate": round(float(group["wrong"].mean()), 4),
        "refusal_rate": round(float(group["refused"].mean()), 4),
        "mean_first_token_prob": round(float(group["first_token_prob"].mean()), 4),
    }


def error_concentration(labels: pd.DataFrame) -> pd.DataFrame:
    """Error rate by answer type, question length, and whether the model answered."""
    rows: list[dict] = []
    for dataset, frame in labels.groupby("dataset"):
        rows.append({"dataset": dataset, **_rate_row("overall", "all questions", frame)})
        for kind, group in frame.groupby("answer_type"):
            rows.append({"dataset": dataset, **_rate_row("answer_type", str(kind), group)})
        buckets = pd.cut(
            frame["n_question_words"],
            bins=list(QUESTION_LENGTH_BINS),
            labels=list(QUESTION_LENGTH_LABELS),
            right=True,
        )
        for bucket, group in frame.groupby(buckets, observed=True):
            rows.append({"dataset": dataset, **_rate_row("question_words", str(bucket), group)})
        for refused, group in frame.groupby("refused"):
            label = "refused" if refused else "committed to an answer"
            rows.append({"dataset": dataset, **_rate_row("refusal", label, group)})
        for rule, group in frame.groupby("match_rule"):
            rows.append({"dataset": dataset, **_rate_row("match_rule", str(rule), group)})
    frame = pd.DataFrame(rows)
    frame["dataset"] = frame["dataset"].fillna("")
    return frame[
        ["dataset", "grouping", "value", "n", "error_rate", "refusal_rate", "mean_first_token_prob"]
    ].reset_index(drop=True)


def _percentile(values: np.ndarray) -> np.ndarray:
    """Rank each value as a percentile in [0, 1], averaging ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks / max(values.size - 1, 1)


def disagreements(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    probe_column: str,
    confidence_column: str,
    split: str,
    n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrong answers where the probe and the confidence signal disagree most.

    Returns (probe caught it, confidence caught it): the first are wrong answers
    the probe ranked as likely wrong while the confidence signal did not, the
    second the reverse.
    """
    frame = scores[scores["split"] == split].copy()
    if frame.empty:
        empty = pd.DataFrame()
        return empty, empty
    frame["probe_pct"] = _percentile(frame[probe_column].to_numpy(dtype=float))
    frame["conf_pct"] = _percentile(frame[confidence_column].to_numpy(dtype=float))
    frame["gap"] = frame["probe_pct"] - frame["conf_pct"]

    wrong = frame[frame["wrong"] == 1].merge(
        labels[["qid", "question", "answer", "matched_alias", "first_token_prob"]],
        on="qid",
        how="left",
    )

    def pick(ascending: bool) -> pd.DataFrame:
        chosen = wrong.sort_values("gap", ascending=ascending).head(n)
        return chosen[
            [
                "qid",
                "question",
                "answer",
                "probe_pct",
                "conf_pct",
                "gap",
                "first_token_prob",
            ]
        ].round(4)

    return pick(ascending=False), pick(ascending=True)


def gold_answers(labels: pd.DataFrame, questions: pd.DataFrame, qids) -> dict[str, str]:
    """A readable gold answer for each question id, for the examples table."""
    out = {}
    for qid in qids:
        aliases = list(questions.loc[qid, "aliases"])
        out[qid] = aliases[0] if aliases else ""
    return out
