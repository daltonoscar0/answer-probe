"""Metrics against values worked out by hand on toy inputs."""

import numpy as np
import pytest

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


def test_auroc_hand_computed():
    # pos scores {0.35, 0.8} vs neg {0.1, 0.4}: 3 of the 4 pairs are ordered.
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_auroc_perfect_reversed_and_chance():
    assert auroc([0.0, 1.0], [0, 1]) == pytest.approx(1.0)
    assert auroc([1.0, 0.0], [0, 1]) == pytest.approx(0.0)
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_auroc_counts_a_tie_as_half_a_pair():
    # pos {1} vs neg {1, 0}: one tie (0.5) plus one win (1.0), over 2 pairs.
    assert auroc([1.0, 1.0, 0.0], [1, 0, 0]) == pytest.approx(0.75)


def test_auroc_is_nan_without_both_classes():
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))


def test_auprc_hand_computed():
    # Sorted by score: 1, 0, 1, 0.  AP = 1*(1/2) + (2/3)*(1/2) = 5/6.
    scores = np.array([0.8, 0.4, 0.35, 0.1])
    labels = np.array([1, 0, 1, 0])
    assert auprc(scores, labels) == pytest.approx(5 / 6)


def test_auprc_perfect_separation_is_one():
    assert auprc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auprc_of_a_constant_score_is_the_base_rate():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.full(5, 0.3)
    assert auprc(scores, labels) == pytest.approx(baserate(labels))
    assert baserate(labels) == pytest.approx(0.4)


def test_risk_coverage_hand_computed():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1])
    coverage, risk = risk_coverage(scores, labels)
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.5, 1 / 3, 0.5])


def test_accuracy_at_coverage_hand_computed():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1])
    assert accuracy_at_coverage(scores, labels, 0.25) == pytest.approx(1.0)
    assert accuracy_at_coverage(scores, labels, 0.5) == pytest.approx(0.5)
    assert accuracy_at_coverage(scores, labels, 1.0) == pytest.approx(0.5)


def test_accuracy_at_coverage_rewards_a_good_signal():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = labels.astype(float)  # high exactly where the answer is wrong
    assert accuracy_at_coverage(perfect, labels, 0.5) == pytest.approx(1.0)


def test_a_constant_signal_scores_the_base_rate_at_every_coverage():
    """Ties are credited with their group's error rate, not the file order.

    The labels here are sorted, so breaking ties by position would hand a
    constant signal a perfect score at 50% coverage.
    """
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    useless = np.zeros(8)
    assert accuracy_at_coverage(useless, labels, 0.5) == pytest.approx(0.5)
    assert accuracy_at_coverage(useless, labels, 0.25) == pytest.approx(0.5)
    _, risk = risk_coverage(useless, labels)
    np.testing.assert_allclose(risk, np.full(8, 0.5))


def test_partially_tied_signal_splits_the_tie_group():
    # Scores 1 and 2 are distinct; the three 3s hold two wrong answers.
    scores = np.array([1.0, 2.0, 3.0, 3.0, 3.0])
    labels = np.array([0, 0, 1, 1, 0])
    # Answering 4 of 5: both distinct-score questions (no errors) plus two of
    # the three tied ones, which carry 2/3 of an error each.
    assert accuracy_at_coverage(scores, labels, 0.8) == pytest.approx(1 - (4 / 3) / 4)


def test_accuracy_at_coverage_rejects_bad_coverage():
    with pytest.raises(ValueError):
        accuracy_at_coverage([0.1, 0.2], [0, 1], 0.0)
    with pytest.raises(ValueError):
        accuracy_at_coverage([0.1, 0.2], [0, 1], 1.5)


def test_expected_calibration_error_hand_computed():
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    # Two bins, each holding half the data and off by 0.1.
    assert expected_calibration_error(probs, labels) == pytest.approx(0.1)


def test_expected_calibration_error_is_zero_when_calibrated():
    probs = np.full(4, 0.25)
    labels = np.array([0, 0, 0, 1])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_expected_calibration_error_is_worst_when_confidently_wrong():
    probs = np.array([1.0, 1.0])
    labels = np.array([0, 0])
    assert expected_calibration_error(probs, labels) == pytest.approx(1.0)


def test_reliability_bins_cover_every_point():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=200)
    labels = (rng.uniform(size=200) < probs).astype(int)
    rel = reliability(probs, labels, n_bins=10)
    assert rel.count.sum() == 200
    assert rel.bin_centre.shape == (10,)
    filled = rel.count > 0
    assert np.all(rel.mean_predicted[filled] >= 0)
    assert np.all(rel.observed[filled] <= 1)


def test_bootstrap_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=400)
    scores = labels + rng.normal(scale=1.0, size=400)
    first = bootstrap(auroc, scores, labels, n_boot=200)
    again = bootstrap(auroc, scores, labels, n_boot=200)
    assert (first.value, first.low, first.high) == (again.value, again.low, again.high)
    assert first.low < first.value < first.high
    assert first.value == pytest.approx(auroc(scores, labels))


def test_bootstrap_interval_is_tighter_with_more_data():
    rng = np.random.default_rng(2)

    def spread(n):
        labels = rng.integers(0, 2, size=n)
        scores = labels + rng.normal(scale=1.5, size=n)
        ci = bootstrap(auroc, scores, labels, n_boot=200)
        return ci.high - ci.low

    assert spread(2000) < spread(200)


def test_bootstrap_difference_of_a_signal_with_itself_is_zero():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=300)
    scores = labels + rng.normal(size=300)
    ci = bootstrap_difference(auroc, scores, scores, labels, n_boot=100)
    assert ci.value == pytest.approx(0.0)
    assert ci.low == pytest.approx(0.0) and ci.high == pytest.approx(0.0)


def test_bootstrap_difference_detects_a_better_signal():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, size=800)
    good = labels + rng.normal(scale=0.5, size=800)
    poor = labels + rng.normal(scale=4.0, size=800)
    ci = bootstrap_difference(auroc, good, poor, labels, n_boot=300)
    assert ci.value > 0
    assert ci.low > 0, "a clearly better signal should have an interval clear of zero"


def test_metrics_reject_mismatched_or_non_binary_input():
    with pytest.raises(ValueError):
        auroc([0.1, 0.2, 0.3], [0, 1])
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [0, 2])
