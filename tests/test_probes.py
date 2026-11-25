"""Probes on synthetic data with a planted direction.

The point of the shuffled-label test is not that it passes here, but that the
same control is run on the real data: if a probe trained on permuted labels
scores above chance, something has leaked between the splits.
"""

import numpy as np
import pytest

from answer_probe.metrics import auroc
from answer_probe.probes import (
    DiffOfMeansProbe,
    Standardizer,
    fit_linear_probe,
    select_linear_probe,
    shuffle_labels,
)

DIM = 20


def synthetic(n, *, seed, strength=2.0):
    """Hidden states whose first few directions carry the error signal."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(DIM, dtype=np.float32)
    direction[:3] = [1.0, -0.5, 0.25]
    direction /= np.linalg.norm(direction)
    x = rng.normal(size=(n, DIM)).astype(np.float32)
    # Give the features different scales, so standardisation has work to do.
    x *= np.linspace(0.5, 5.0, DIM, dtype=np.float32)
    logits = strength * (x / np.linspace(0.5, 5.0, DIM, dtype=np.float32)) @ direction
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return x, y, direction


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_linear_probe_recovers_the_planted_direction():
    x, y, direction = synthetic(4000, seed=0)
    probe = fit_linear_probe(x, y, l2=1e-4)
    assert cosine(probe.weights, direction) > 0.95
    assert auroc(probe.score(x), y) > 0.85


def test_linear_probe_generalises_to_held_out_data():
    x, y, _ = synthetic(4000, seed=1)
    x_test, y_test, _ = synthetic(2000, seed=2)
    probe = fit_linear_probe(x, y, l2=1e-3)
    assert auroc(probe.score(x_test), y_test) > 0.85


def test_shuffled_labels_score_near_chance_on_held_out_data():
    """Averaged over permutations, because one permutation is noisy.

    A probe fitted to permuted labels lands on a direction that is noise, but a
    noise direction in D dimensions still aligns with the real one by about
    1/sqrt(D) by chance; at D=20 single permutations reach AUROC 0.63 and the
    spread across permutations is about 0.065. The pipeline reports the mean and
    the range over several permutations for the same reason.
    """
    x, y, _ = synthetic(4000, seed=3)
    x_test, y_test, _ = synthetic(2000, seed=4)
    scores = [
        auroc(fit_linear_probe(x, shuffle_labels(y, seed=s), l2=1e-3).score(x_test), y_test)
        for s in range(20)
    ]
    assert float(np.mean(scores)) == pytest.approx(0.5, abs=0.05)


def test_shuffled_labels_still_fit_the_training_split():
    """The control has to be checked out of sample.

    With 20 features and 300 rows a probe can partly memorise permuted labels,
    which is exactly why the reported control is measured on held-out data.
    """
    x, y, _ = synthetic(300, seed=5)
    shuffled = shuffle_labels(y, seed=8)
    probe = fit_linear_probe(x, shuffled, l2=1e-4)
    assert auroc(probe.score(x), shuffled) > 0.6


def test_diff_of_means_recovers_the_direction_without_training():
    x, y, direction = synthetic(4000, seed=6)
    probe = DiffOfMeansProbe.fit(x, y)
    assert cosine(probe.direction, direction) > 0.9
    assert auroc(probe.score(x), y) > 0.8
    assert np.linalg.norm(probe.direction) == pytest.approx(1.0, abs=1e-5)


def test_diff_of_means_shuffled_control_is_near_chance():
    x, y, _ = synthetic(4000, seed=9)
    x_test, y_test, _ = synthetic(2000, seed=10)
    scores = [
        auroc(DiffOfMeansProbe.fit(x, shuffle_labels(y, seed=s)).score(x_test), y_test)
        for s in range(20)
    ]
    assert float(np.mean(scores)) == pytest.approx(0.5, abs=0.05)


def test_diff_of_means_needs_both_classes():
    x, _, _ = synthetic(50, seed=12)
    with pytest.raises(ValueError):
        DiffOfMeansProbe.fit(x, np.ones(50, dtype=int))


def test_regularisation_is_chosen_on_validation_only():
    x, y, _ = synthetic(2000, seed=13)
    x_val, y_val, _ = synthetic(1000, seed=14)
    probe, trace = select_linear_probe(x, y, x_val, y_val)
    assert len(trace) == 6
    assert probe.l2 == max(trace, key=lambda row: row["val_score"])["l2"]
    # The chosen probe's standardiser comes from training data alone.
    np.testing.assert_allclose(probe.standardizer.mean, x.mean(axis=0), rtol=1e-5)


def test_heavy_regularisation_shrinks_the_weights():
    x, y, _ = synthetic(2000, seed=15)
    weak = fit_linear_probe(x, y, l2=1e-4)
    strong = fit_linear_probe(x, y, l2=10.0)
    assert np.linalg.norm(strong.weights) < np.linalg.norm(weak.weights)


def test_probabilities_are_in_range_and_monotone_in_the_score():
    x, y, _ = synthetic(1000, seed=16)
    probe = fit_linear_probe(x, y, l2=1e-3)
    probs = probe.predict_proba(x)
    assert np.all((probs > 0) & (probs < 1))
    order_score = np.argsort(probe.score(x))
    order_prob = np.argsort(probs)
    np.testing.assert_array_equal(order_score, order_prob)


def test_standardizer_handles_a_constant_feature():
    x = np.ones((10, 3), dtype=np.float32)
    x[:, 1] = np.arange(10)
    std = Standardizer.fit(x)
    z = std.apply(x)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z[:, 0], 0.0)
    assert z[:, 1].std() == pytest.approx(1.0, abs=1e-5)


def test_training_is_deterministic():
    x, y, _ = synthetic(500, seed=17)
    first = fit_linear_probe(x, y, l2=1e-3)
    again = fit_linear_probe(x, y, l2=1e-3)
    np.testing.assert_array_equal(first.weights, again.weights)
    assert first.bias == again.bias


def test_a_single_class_split_is_reported_clearly():
    """A probe cannot be fitted when every answer is wrong; say so plainly."""
    x, y, _ = synthetic(200, seed=18)
    x_val, y_val, _ = synthetic(100, seed=19)
    with pytest.raises(ValueError, match="only one class"):
        select_linear_probe(x, np.ones_like(y), x_val, y_val)
    with pytest.raises(ValueError, match="only one class"):
        select_linear_probe(x, y, x_val, np.zeros_like(y_val))
