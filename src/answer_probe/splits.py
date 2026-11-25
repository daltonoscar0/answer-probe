"""Question-level splits.

TriviaQA is cut 70/15/15 into train/val/test. The shift set (NQ-open) is a
fourth split that never enters training or model selection: it exists only to
ask whether what a probe learned on TriviaQA transfers.

Splits are keyed on question id, so the same question can never land in two
splits, and the assignment does not depend on the order rows were read in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 20260103
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

TRAIN, VAL, TEST, SHIFT = "train", "val", "test", "shift"
SPLIT_NAMES = (TRAIN, VAL, TEST, SHIFT)

# Which splits get the expensive 8-sample self-consistency pass. The baseline is
# only ever *evaluated* on test and shift; val is included so the two-feature
# combination model can be fitted without touching test.
SELFCONSISTENCY_SPLITS = (VAL, TEST, SHIFT)


def make_splits(
    qids: list[str],
    datasets: list[str],
    *,
    main_dataset: str = "triviaqa",
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Assign every question id to exactly one split."""
    if len(qids) != len(datasets):
        raise ValueError("qids and datasets must be the same length")
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate question ids")

    main = sorted(q for q, d in zip(qids, datasets, strict=True) if d == main_dataset)
    shift = sorted(q for q, d in zip(qids, datasets, strict=True) if d != main_dataset)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(main))
    n_train = int(round(FRACTIONS["train"] * len(main)))
    n_val = int(round(FRACTIONS["val"] * len(main)))
    cuts = {
        TRAIN: order[:n_train],
        VAL: order[n_train : n_train + n_val],
        TEST: order[n_train + n_val :],
    }
    splits = {name: sorted(main[i] for i in idx) for name, idx in cuts.items()}
    splits[SHIFT] = shift
    check_splits(splits)
    return splits


def check_splits(splits: dict[str, list[str]]) -> None:
    """Raise unless the splits are a partition."""
    seen: set[str] = set()
    for name, qids in splits.items():
        overlap = seen & set(qids)
        if overlap:
            raise ValueError(f"split {name!r} shares {len(overlap)} questions with another split")
        if len(set(qids)) != len(qids):
            raise ValueError(f"split {name!r} contains duplicate question ids")
        seen |= set(qids)


def selfconsistency_qids(splits: dict[str, list[str]]) -> list[str]:
    """The questions that get the 8-sample self-consistency pass."""
    out: list[str] = []
    for name in SELFCONSISTENCY_SPLITS:
        out.extend(splits.get(name, []))
    return sorted(out)


def save_splits(splits: dict[str, list[str]], path: str | Path, *, seed: int = SPLIT_SEED) -> None:
    check_splits(splits)
    payload = {
        "seed": seed,
        "fractions": FRACTIONS,
        "counts": {name: len(qids) for name, qids in splits.items()},
        "selfconsistency_splits": list(SELFCONSISTENCY_SPLITS),
        "splits": {name: list(qids) for name, qids in splits.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    splits = payload["splits"]
    check_splits(splits)
    return splits


def split_of(splits: dict[str, list[str]]) -> dict[str, str]:
    """Invert the split map: question id -> split name."""
    return {qid: name for name, qids in splits.items() for qid in qids}
