"""Build a small synthetic extraction run, with the schema the notebook writes.

Used by the end-to-end test and by `scripts/smoke.py`. The hidden states carry a
planted error direction that grows with depth, so the layer curve has a shape
and the pipeline has something real to find; the confidence signals carry a
weaker version of the same signal, so the probe-versus-confidence comparison is
not degenerate.

Nothing here is a result. It exists to prove the pipeline computes what it says
it computes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from answer_probe import io

N_LAYERS = 6
HIDDEN = 24

ANSWERS_RIGHT = ["Paris", "1969", "The Beatles", "nineteen sixty nine", "Mount Everest"]
ANSWERS_WRONG = ["London", "1972", "The Rolling Stones", "unknown", "K2"]
QUESTION_TEMPLATES = [
    "Who wrote the novel number {i}?",
    "In what year did event number {i} happen?",
    "Which city hosted contest number {i}?",
    "How many parts does object number {i} have?",
    "Where is landmark number {i}?",
    "Name the author of work number {i}.",
]


def build(
    *,
    n_main: int = 400,
    n_shift: int = 150,
    seed: int = 0,
    probe_strength: float = 1.6,
    confidence_strength: float = 0.9,
) -> dict:
    """Return questions, generations, self-consistency rows and activations."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=HIDDEN)
    direction /= np.linalg.norm(direction)

    questions, generations, selfcons = [], [], []
    qids: list[str] = []
    datasets: list[str] = []
    wrong_flags: list[int] = []

    for dataset, count in (("triviaqa", n_main), ("nq_open", n_shift)):
        for i in range(count):
            qid = f"{dataset}:{i}"
            # The shift set is harder, as a distribution shift should be.
            p_wrong = 0.62 if dataset == "triviaqa" else 0.78
            is_wrong = int(rng.uniform() < p_wrong)
            gold = ANSWERS_RIGHT[i % len(ANSWERS_RIGHT)]
            questions.append(
                {
                    "qid": qid,
                    "dataset": dataset,
                    "question": QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)].format(i=i),
                    "aliases": [gold] if gold != "nineteen sixty nine" else [gold, "1969"],
                }
            )
            qids.append(qid)
            datasets.append(dataset)
            wrong_flags.append(is_wrong)

    wrong = np.asarray(wrong_flags, dtype=float)
    n = len(qids)

    # Confidence signals: informative, but less so than the deepest layers.
    noise = rng.normal(size=n)
    logit_conf = confidence_strength * (0.5 - wrong) + 0.8 * noise
    first_token_prob = 1 / (1 + np.exp(-logit_conf))
    entropy = 2.5 * (1 - first_token_prob) + 0.2 * np.abs(rng.normal(size=n))
    answer_logprob = -0.4 - 1.6 * (1 - first_token_prob) + 0.1 * rng.normal(size=n)

    for i, qid in enumerate(qids):
        generations.append(
            {
                "qid": qid,
                "answer": (
                    ANSWERS_WRONG[i % len(ANSWERS_WRONG)]
                    if wrong_flags[i]
                    else ANSWERS_RIGHT[i % len(ANSWERS_RIGHT)]
                ),
                "n_answer_tokens": int(rng.integers(1, 6)),
                "first_token_prob": float(first_token_prob[i]),
                "first_token_entropy": float(entropy[i]),
                "answer_logprob_mean": float(answer_logprob[i]),
                "answer_logprob_sum": float(answer_logprob[i] * 3),
            }
        )

    # Self-consistency on the evaluation splits only, as in the real run.
    from answer_probe.splits import make_splits, selfconsistency_qids

    split_map = make_splits(qids, datasets)
    wanted = set(selfconsistency_qids(split_map))
    agreement_rate = 0.85 - 0.5 * wrong
    for i, qid in enumerate(qids):
        if qid not in wanted:
            continue
        agree = int(rng.binomial(8, np.clip(agreement_rate[i], 0.02, 0.98)))
        selfcons.append({"qid": qid, "agreement": agree / 8, "n_samples": 8})

    # Activations: the planted direction strengthens with depth, and the
    # after-answering position carries it more strongly than before answering.
    before = np.zeros((n, N_LAYERS, HIDDEN), dtype=np.float32)
    after = np.zeros((n, N_LAYERS, HIDDEN), dtype=np.float32)
    scale = np.linspace(0.5, 4.0, HIDDEN)
    for layer in range(N_LAYERS):
        depth = layer / (N_LAYERS - 1)
        for position, array, boost in (("before", before, 1.0), ("after", after, 1.35)):
            del position
            base = rng.normal(size=(n, HIDDEN))
            signal = probe_strength * boost * depth * (wrong - 0.5)[:, None] * direction[None, :]
            array[:, layer, :] = (base + signal) * scale[None, :]

    return {
        "qids": qids,
        "questions": questions,
        "generations": generations,
        "selfconsistency": selfcons,
        "before": before.astype(np.float16),
        "after": after.astype(np.float16),
        "split_map": split_map,
        "wrong": wrong.astype(int),
    }


def write(root: str | Path, **kwargs) -> Path:
    """Write a synthetic run to `root` and return the path."""
    payload = build(**kwargs)
    summary = pd.DataFrame(
        [
            {
                "dataset": "triviaqa",
                "hf_path": "SYNTHETIC",
                "split": "n/a",
                "n_sampled": sum(1 for q in payload["questions"] if q["dataset"] == "triviaqa"),
                "license": "n/a (synthetic fixture)",
            },
            {
                "dataset": "nq_open",
                "hf_path": "SYNTHETIC",
                "split": "n/a",
                "n_sampled": sum(1 for q in payload["questions"] if q["dataset"] == "nq_open"),
                "license": "n/a (synthetic fixture)",
            },
        ]
    )
    return io.write_run_folder(
        root,
        manifest={
            "synthetic": True,
            "model": "synthetic-fixture",
            "checkpoint": "none",
            "prompt_version": "fixture",
            "note": "Generated by tests/fake_run.py. Not a result.",
        },
        questions=payload["questions"],
        generations=payload["generations"],
        selfconsistency=payload["selfconsistency"],
        before=payload["before"],
        after=payload["after"],
        qids=payload["qids"],
        data_summary=summary,
        split_map=payload["split_map"],
    )
