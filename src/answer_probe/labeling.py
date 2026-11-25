"""Turning generated answers into correctness labels, and checking those labels
against a human grader.

The automatic labels drive every number in the study, so they get audited: 60
answers are drawn stratified by automatic label, graded by hand without the
automatic label visible, and the agreement is reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from answer_probe.features import answer_type, gold_is_number
from answer_probe.io import Run, data_dir, index_by_qid, read_jsonl
from answer_probe.normalize import is_refusal, match_answer


def human_labels_path() -> Path:
    return data_dir() / "human_labels.jsonl"


def grading_sheet_path() -> Path:
    return data_dir() / "human_grading_sheet.jsonl"

GRADE_SAMPLE_SIZE = 60
GRADE_SAMPLE_SEED = 20260105
AGREEMENT_THRESHOLD = 0.90


def build_labels(run: Run, splits: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """One row per question: the answer, whether it was right, and every signal."""
    questions = run.questions
    generations = run.generations
    missing = set(questions["qid"]) - set(generations["qid"])
    if missing:
        raise ValueError(
            f"{len(missing)} sampled questions have no generation; "
            "the notebook run is incomplete"
        )

    selfcons = run.selfconsistency
    agreement = (
        dict(zip(selfcons["qid"], selfcons["agreement"], strict=True))
        if len(selfcons)
        else {}
    )
    split_of = (
        {qid: name for name, qids in splits.items() for qid in qids} if splits else {}
    )

    rows = []
    for qid, question_row in questions.iterrows():
        generation = generations.loc[qid]
        answer = str(generation["answer"])
        aliases = list(question_row["aliases"])
        match = match_answer(answer, aliases)
        rows.append(
            {
                "qid": qid,
                "dataset": question_row["dataset"],
                "split": split_of.get(qid, ""),
                "question": question_row["question"],
                "answer": answer,
                "n_aliases": len(aliases),
                "correct": int(match.correct),
                "wrong": int(not match.correct),
                "match_rule": match.rule,
                "matched_alias": match.alias or "",
                "refused": int(is_refusal(answer)),
                "answer_type": answer_type(question_row["question"]),
                "gold_is_number": int(gold_is_number(aliases)),
                "n_question_words": len(str(question_row["question"]).split()),
                "n_answer_tokens": int(generation["n_answer_tokens"]),
                "first_token_prob": float(generation["first_token_prob"]),
                "first_token_entropy": float(generation["first_token_entropy"]),
                "answer_logprob_mean": float(generation["answer_logprob_mean"]),
                "answer_logprob_sum": float(generation["answer_logprob_sum"]),
                "selfconsistency_agreement": float(agreement.get(qid, np.nan)),
            }
        )
    return index_by_qid(pd.DataFrame(rows))


def accuracy_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and refusal rate per dataset, and per split within a dataset."""
    rows = []
    for (dataset, split), group in labels.groupby(["dataset", "split"], dropna=False):
        rows.append(_accuracy_row(dataset, split or "(all)", group))
    for dataset, group in labels.groupby("dataset"):
        rows.append(_accuracy_row(dataset, "all", group))
    return pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)


def _accuracy_row(dataset: str, split: str, group: pd.DataFrame) -> dict:
    return {
        "dataset": dataset,
        "split": split,
        "n": len(group),
        "accuracy": round(float(group["correct"].mean()), 4),
        "error_rate": round(float(group["wrong"].mean()), 4),
        "refusal_rate": round(float(group["refused"].mean()), 4),
        "exact_match_share": round(float((group["match_rule"] == "exact").mean()), 4),
        "span_match_share": round(float((group["match_rule"] == "span").mean()), 4),
    }


def grading_sample(
    labels: pd.DataFrame,
    *,
    n: int = GRADE_SAMPLE_SIZE,
    seed: int = GRADE_SAMPLE_SEED,
) -> pd.DataFrame:
    """Draw `n` answers stratified by automatic label, for hand grading.

    Stratified so both halves of the matcher get audited: an automatic scorer
    can be wrong by calling a right answer wrong just as easily as the reverse,
    and the wrong class is much the larger one.
    """
    rng = np.random.default_rng(seed)
    # `labels` carries qid as both the index and a column; drop the index so
    # sorting by question id is unambiguous.
    labels = labels.reset_index(drop=True)
    per_class = n // 2
    chunks = []
    for value in (1, 0):
        pool = labels[labels["correct"] == value].sort_values("qid")
        take = min(per_class, len(pool))
        if take < len(pool):
            picked = rng.choice(len(pool), size=take, replace=False)
            chunks.append(pool.iloc[np.sort(picked)])
        else:
            chunks.append(pool)
    sample = pd.concat(chunks)
    # Shuffle so the grader cannot infer the automatic label from the order.
    order = rng.permutation(len(sample))
    return sample.iloc[order].reset_index(drop=True)


def write_grading_sheet(
    sample: pd.DataFrame, questions: pd.DataFrame, path: str | Path | None = None
) -> Path:
    """Write the grading sheet, deliberately without the automatic label."""
    path = Path(path) if path else grading_sheet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for _, row in sample.iterrows():
            handle.write(
                json.dumps(
                    {
                        "qid": row["qid"],
                        "dataset": row["dataset"],
                        "question": row["question"],
                        "model_answer": row["answer"],
                        "gold_answers": list(questions.loc[row["qid"], "aliases"]),
                        "human_correct": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_human_labels(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else human_labels_path()
    if not path.exists():
        return pd.DataFrame(columns=["qid", "human_correct"])
    rows = [r for r in read_jsonl(path) if r.get("human_correct") is not None]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["qid", "human_correct"])
    frame["human_correct"] = frame["human_correct"].astype(int)
    return frame.drop_duplicates(subset="qid", keep="last")


def agreement_report(labels: pd.DataFrame, human: pd.DataFrame) -> dict:
    """Compare the matcher with the hand grades on the audited sample."""
    if human.empty:
        return {"n_graded": 0, "agreement": None}
    merged = human.merge(
        labels[["qid", "correct", "answer", "question", "match_rule"]], on="qid", how="inner"
    )
    if merged.empty:
        return {"n_graded": 0, "agreement": None}
    agree = merged["correct"] == merged["human_correct"]
    auto_right_human_wrong = merged[(merged["correct"] == 1) & (merged["human_correct"] == 0)]
    auto_wrong_human_right = merged[(merged["correct"] == 0) & (merged["human_correct"] == 1)]
    return {
        "n_graded": int(len(merged)),
        "agreement": round(float(agree.mean()), 4),
        "n_disagree": int((~agree).sum()),
        "auto_correct_human_wrong": int(len(auto_right_human_wrong)),
        "auto_wrong_human_correct": int(len(auto_wrong_human_right)),
        "threshold": AGREEMENT_THRESHOLD,
        "meets_threshold": bool(float(agree.mean()) >= AGREEMENT_THRESHOLD),
        "disagreements": [
            {
                "qid": row["qid"],
                "question": row["question"],
                "model_answer": row["answer"],
                "automatic": int(row["correct"]),
                "human": int(row["human_correct"]),
                "match_rule": row["match_rule"],
            }
            for _, row in merged[~agree].iterrows()
        ],
    }
