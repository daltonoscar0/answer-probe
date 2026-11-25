"""Metrics, written out so every number in the README can be checked by hand.

Throughout, the positive class is "the answer is wrong". A score is a signal
that should be *high* when the answer is wrong, so confidence signals are
negated before they get here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260104


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} disagree")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be 0/1")
    return scores, labels


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney) identity.

    Ties get averaged ranks, so a constant score gives exactly 0.5.
    """
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision: the step-wise sum of precision x recall increments.

    This is the `average_precision_score` definition, not the trapezoidal
    interpolation of a precision-recall curve.
    """
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    # Only thresholds at the end of a run of equal scores are real operating
    # points; inside a tie the ordering is arbitrary.
    last_of_run = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    tp, fp = tp[last_of_run], fp[last_of_run]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_increments = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * recall_increments))


def baserate(labels: np.ndarray) -> float:
    """The share of wrong answers, which is the AUPRC a random score gets."""
    labels = np.asarray(labels).ravel().astype(int)
    return float(labels.mean())


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap percentile interval."""

    value: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {prefix: self.value, f"{prefix}_low": self.low, f"{prefix}_high": self.high}


def bootstrap(
    fn,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Resample *questions* with replacement and take percentile bounds.

    Resampling questions (not predictions) is the right unit here: every
    question contributes exactly one score and one label, and the intervals have
    to reflect which questions we happened to sample.
    """
    scores, labels = _check(scores, labels)
    point = fn(scores, labels)
    rng = np.random.default_rng(seed)
    n = scores.size
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = fn(scores[idx], labels[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return Interval(point, float("nan"), float("nan"))
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(low), float(high))


def bootstrap_difference(
    fn,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Paired bootstrap of `fn(a) - fn(b)` on the same resampled questions.

    Paired, because comparing two independent intervals understates how much of
    the variation the two signals share.
    """
    scores_a, labels = _check(scores_a, labels)
    scores_b, _ = _check(scores_b, labels)
    point = fn(scores_a, labels) - fn(scores_b, labels)
    rng = np.random.default_rng(seed)
    n = labels.size
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = fn(scores_a[idx], labels[idx]) - fn(scores_b[idx], labels[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return Interval(point, float("nan"), float("nan"))
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(low), float(high))


def _expected_errors(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cumulative expected errors when answering in increasing order of score.

    Within a run of equal scores there is no reason to prefer one question over
    another, so each tied question is credited with the error rate of its tie
    group. Without this, a signal with few distinct values -- self-consistency
    takes only nine, 0/8 to 8/8 -- would score according to the order the
    questions happen to sit in the file.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)

    expected = sorted_labels.copy()
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            expected[start:i] = sorted_labels[start:i].mean()
            start = i
    return np.cumsum(expected)


def risk_coverage(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The risk-coverage curve for abstaining on the highest scores.

    `scores` is high when the answer is predicted wrong, so the system answers
    the lowest-scoring questions first. Returns (coverage, risk) where risk is
    the error rate among the answered questions, for coverage 1/n .. 1.
    """
    scores, labels = _check(scores, labels)
    n = scores.size
    answered = np.arange(1, n + 1)
    return answered / n, _expected_errors(scores, labels) / answered


def accuracy_at_coverage(
    scores: np.ndarray, labels: np.ndarray, coverage: float
) -> float:
    """Accuracy among answered questions when answering `coverage` of them."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    scores, labels = _check(scores, labels)
    n_answer = max(1, int(round(coverage * scores.size)))
    errors = _expected_errors(scores, labels)[n_answer - 1]
    return float(1.0 - errors / n_answer)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    """Equal-width-bin ECE between predicted P(wrong) and observed frequency."""
    probs, labels = _check(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `digitize` with right=True puts 0.0 in bin 0 rather than a bin of its own.
    bins = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        total += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(total)


@dataclass(frozen=True)
class Reliability:
    """Binned reliability, for the calibration plot."""

    bin_centre: np.ndarray
    mean_predicted: np.ndarray
    observed: np.ndarray
    count: np.ndarray


def reliability(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> Reliability:
    probs, labels = _check(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    centres, predicted, observed, counts = [], [], [], []
    for b in range(n_bins):
        mask = bins == b
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        counts.append(int(mask.sum()))
        predicted.append(float(probs[mask].mean()) if mask.any() else np.nan)
        observed.append(float(labels[mask].mean()) if mask.any() else np.nan)
    return Reliability(
        np.array(centres), np.array(predicted), np.array(observed), np.array(counts)
    )
