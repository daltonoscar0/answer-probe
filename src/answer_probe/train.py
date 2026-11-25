"""Fit every probe and every baseline, and write their per-question scores.

`train` does all the fitting and model selection and saves the scores;
`evaluate` only reads those scores and computes metrics. Keeping them apart
means the reported numbers can be recomputed from `results/scores.csv` without
any model being refitted, and selection cannot accidentally see the test split.

Everything here is oriented so a larger score means "this answer is wrong".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from answer_probe import io
from answer_probe.features import (
    CONFIDENCE_SIGNALS,
    SURFACE_FEATURE_NAMES,
    confidence_score,
    surface_features,
)
from answer_probe.metrics import auroc
from answer_probe.probes import (
    L2_GRID,
    DiffOfMeansProbe,
    fit_linear_probe,
    select_linear_probe,
    shuffle_labels,
)
from answer_probe.splits import SHIFT, TEST, TRAIN, VAL

PROBE_KINDS = ("linear", "diffmeans")
# Penalty for the two-feature combination model (see `fit_combination`).
COMBINATION_L2 = 1e-3
N_SHUFFLE_SEEDS = 5
SHUFFLE_SEED_BASE = 20260106
EVAL_SPLITS = (VAL, TEST, SHIFT)


@dataclass
class Dataset:
    """Labels and split membership, in a fixed question order per split."""

    labels: pd.DataFrame
    splits: dict[str, list[str]]

    def qids(self, split: str) -> list[str]:
        return [q for q in self.splits[split] if q in self.labels.index]

    def y(self, split: str) -> np.ndarray:
        return self.labels.loc[self.qids(split), "wrong"].to_numpy(dtype=int)


def _layer_scores(run: io.Run, data: Dataset, position: str, layer: int) -> dict:
    """Fit both probes at one layer and score every split."""
    x = {split: run.layer(position, layer, data.qids(split)) for split in
         (TRAIN, VAL, TEST, SHIFT)}
    y_train, y_val = data.y(TRAIN), data.y(VAL)

    probe, trace = select_linear_probe(x[TRAIN], y_train, x[VAL], y_val)
    standardizer = probe.standardizer
    dom = DiffOfMeansProbe.fit(x[TRAIN], y_train, standardizer=standardizer)

    return {
        "linear": {
            "probe": probe,
            "l2": probe.l2,
            "trace": trace,
            "scores": {split: probe.score(x[split]) for split in EVAL_SPLITS},
            "probs": {split: probe.predict_proba(x[split]) for split in EVAL_SPLITS},
        },
        "diffmeans": {
            "probe": dom,
            "l2": None,
            "trace": [],
            "scores": {split: dom.score(x[split]) for split in EVAL_SPLITS},
            "probs": {},
        },
    }


def scan_layers(run: io.Run, data: Dataset, *, progress=None) -> tuple[pd.DataFrame, dict]:
    """Fit both probes at every layer and both positions.

    Returns the per-layer validation/test/shift AUROCs and, keyed by
    (position, layer, kind), the fitted objects and their scores.
    """
    rows = []
    fitted: dict[tuple[str, int, str], dict] = {}
    total = len(io.POSITIONS) * run.n_layers
    done = 0
    for position in io.POSITIONS:
        for layer in range(run.n_layers):
            result = _layer_scores(run, data, position, layer)
            for kind in PROBE_KINDS:
                entry = result[kind]
                fitted[(position, layer, kind)] = entry
                row = {"position": position, "layer": layer, "probe": kind, "l2": entry["l2"]}
                for split in EVAL_SPLITS:
                    row[f"auroc_{split}"] = auroc(entry["scores"][split], data.y(split))
                rows.append(row)
            done += 1
            if progress:
                progress(done, total, position, layer)
    return pd.DataFrame(rows), fitted


def select_best(scan: pd.DataFrame) -> dict:
    """Pick the best layer per (position, probe) on validation AUROC only."""
    selection: dict[str, dict] = {}
    for (position, kind), group in scan.groupby(["position", "probe"]):
        best = group.loc[group["auroc_val"].idxmax()]
        selection[f"{position}_{kind}"] = {
            "position": position,
            "probe": kind,
            "layer": int(best["layer"]),
            "l2": None if pd.isna(best["l2"]) else float(best["l2"]),
            "auroc_val": float(best["auroc_val"]),
            "selected_on": "val",
        }
    # The project's question is about the state before answering, so that is the
    # headline probe even if the after-answering one scores higher on
    # validation. The best of all four is reported alongside it.
    probe_keys = sorted(selection)
    return {
        "per_probe": selection,
        "headline": dict(selection["before_linear"]),
        "overall_best_key": max(probe_keys, key=lambda k: selection[k]["auroc_val"]),
    }


def shuffled_control(
    run: io.Run, data: Dataset, position: str, layer: int, l2: float
) -> pd.DataFrame:
    """Refit the selected probe on permuted training labels.

    Measured out of sample: a probe with 1152 features can partly memorise
    permuted labels in-sample, so only the held-out score says whether anything
    leaked. Averaged over permutations, because one permutation is noisy.
    """
    x_train = run.layer(position, layer, data.qids(TRAIN))
    x_test = run.layer(position, layer, data.qids(TEST))
    y_train, y_test = data.y(TRAIN), data.y(TEST)
    rows = []
    for i in range(N_SHUFFLE_SEEDS):
        seed = SHUFFLE_SEED_BASE + i
        permuted = shuffle_labels(y_train, seed=seed)
        probe = fit_linear_probe(x_train, permuted, l2=l2)
        dom = DiffOfMeansProbe.fit(x_train, permuted, standardizer=probe.standardizer)
        rows.append(
            {
                "seed": seed,
                "linear_auroc_test": auroc(probe.score(x_test), y_test),
                "diffmeans_auroc_test": auroc(dom.score(x_test), y_test),
                "linear_auroc_train_insample": auroc(probe.score(x_train), permuted),
            }
        )
    return pd.DataFrame(rows)


def fit_surface_baseline(data: Dataset) -> dict[str, np.ndarray]:
    """Logistic regression on features of the question text alone."""
    labels = data.labels

    def design(split):
        rows = labels.loc[data.qids(split)]
        return surface_features(list(rows["question"]), [list(a) for a in rows["aliases"]])

    x_train, y_train = design(TRAIN), data.y(TRAIN)
    x_val, y_val = design(VAL), data.y(VAL)
    probe, trace = select_linear_probe(x_train, y_train, x_val, y_val)
    return {
        "scores": {split: probe.score(design(split)) for split in EVAL_SPLITS},
        "l2": probe.l2,
        "trace": trace,
        "weights": dict(zip(SURFACE_FEATURE_NAMES, probe.weights.tolist(), strict=True)),
    }


def available_confidence_signals(data: Dataset) -> list[str]:
    """Confidence signals that have a value for every question we evaluate on."""
    out = []
    for name, spec in CONFIDENCE_SIGNALS.items():
        column = spec["column"]
        if column not in data.labels.columns:
            continue
        rows = data.labels.loc[
            [q for split in EVAL_SPLITS for q in data.qids(split)], column
        ]
        if rows.notna().all():
            out.append(name)
    return out


def best_confidence_signal(data: Dataset, names: list[str]) -> str:
    """The confidence baseline with the highest validation AUROC."""
    y_val = data.y(VAL)
    scored = {
        name: auroc(confidence_score(data.labels.loc[data.qids(VAL)], name), y_val)
        for name in names
    }
    return max(scored, key=lambda name: scored[name])


def fit_combination(
    data: Dataset,
    probe_scores: dict[str, np.ndarray],
    confidence_name: str,
) -> dict:
    """A two-feature logistic model on (probe score, best confidence signal).

    Fitted on validation, which is also where the probe's layer and the
    confidence signal were chosen, so the test split stays untouched.
    """
    def design(split):
        conf = confidence_score(data.labels.loc[data.qids(split)], confidence_name)
        return np.column_stack([probe_scores[split], conf]).astype(np.float32)

    # A fixed light penalty rather than a grid search: with two features there
    # is nothing to select, and selecting on validation after fitting on
    # validation would just pick the smallest penalty every time.
    probe = fit_linear_probe(design(VAL), data.y(VAL), l2=COMBINATION_L2)
    return {
        "scores": {split: probe.score(design(split)) for split in EVAL_SPLITS},
        "l2": COMBINATION_L2,
        "weights": {
            "probe": float(probe.weights[0]),
            confidence_name: float(probe.weights[1]),
        },
        "confidence_signal": confidence_name,
    }


def build_scores_frame(
    data: Dataset,
    fitted: dict,
    selection: dict,
    surface: dict,
    combination: dict,
    confidence_names: list[str],
) -> pd.DataFrame:
    """One row per evaluated question, one column per signal."""
    frames = []
    for split in EVAL_SPLITS:
        qids = data.qids(split)
        rows = data.labels.loc[qids]
        frame = pd.DataFrame(
            {
                "qid": qids,
                "split": split,
                "dataset": rows["dataset"].to_numpy(),
                "wrong": data.y(split),
                "refused": rows["refused"].to_numpy(),
                "answer_type": rows["answer_type"].to_numpy(),
            }
        )
        for key, choice in selection["per_probe"].items():
            entry = fitted[(choice["position"], choice["layer"], choice["probe"])]
            frame[f"probe_{key}"] = entry["scores"][split]
            if entry["probs"]:
                frame[f"prob_{key}"] = entry["probs"][split]
        for name in confidence_names:
            frame[f"conf_{name}"] = confidence_score(rows, name)
        frame["surface"] = surface["scores"][split]
        frame["combination"] = combination["scores"][split]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_training(run: io.Run, labels: pd.DataFrame, splits: dict, *, progress=None) -> dict:
    """The whole fitting stage. Returns the artefacts `evaluate` and `report` read."""
    labels = labels.copy()
    aliases = run.questions["aliases"]
    labels["aliases"] = [list(aliases.loc[qid]) for qid in labels["qid"]]
    data = Dataset(labels=labels, splits=splits)

    scan, fitted = scan_layers(run, data, progress=progress)
    selection = select_best(scan)
    headline = selection["headline"]
    controls = shuffled_control(
        run, data, headline["position"], headline["layer"], headline["l2"] or L2_GRID[0]
    )
    surface = fit_surface_baseline(data)
    confidence_names = available_confidence_signals(data)
    if not confidence_names:
        raise RuntimeError("no confidence signal is complete across the evaluation splits")
    best_conf = best_confidence_signal(data, confidence_names)

    headline_entry = fitted[(headline["position"], headline["layer"], "linear")]
    combination = fit_combination(data, headline_entry["scores"], best_conf)

    scores = build_scores_frame(
        data, fitted, selection, surface, combination, confidence_names
    )
    return {
        "scan": scan,
        "selection": selection,
        "controls": controls,
        "surface": surface,
        "combination": combination,
        "confidence_names": confidence_names,
        "best_confidence": best_conf,
        "scores": scores,
    }
