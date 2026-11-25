"""The probes: an L2-regularised logistic probe trained with Optax, and a
difference-of-means probe that needs no training at all.

Both predict "this answer is wrong" from a hidden state. Features are
standardised using the training split's mean and standard deviation only.

Training is deterministic: the weights start at zero, the bias starts at the
training base rate's logit, and the optimiser sees the whole split at once, so
there is no seed and no shuffling to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

# L2 strengths searched on the validation split.
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
STEPS = 1000
LEARNING_RATE = 0.05


@dataclass(frozen=True)
class Standardizer:
    """Per-feature mean and standard deviation, fitted on training data."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A feature that never varies carries no information; dividing by 1
        # leaves it at zero after centring rather than producing infinities.
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.scale


@dataclass(frozen=True)
class LinearProbe:
    """A fitted logistic probe. `score` is the logit of P(wrong)."""

    weights: np.ndarray
    bias: float
    l2: float
    standardizer: Standardizer
    final_loss: float

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.apply(x)
        return np.asarray(z @ self.weights + self.bias, dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(x), -60, 60)))


def _loss_fn(params, x, y, l2):
    logits = x @ params["w"] + params["b"]
    # Mean binary cross-entropy plus an L2 penalty on the weights only.
    bce = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return bce + l2 * jnp.sum(params["w"] ** 2)


def _fit(x: np.ndarray, y: np.ndarray, l2: float, steps: int, learning_rate: float):
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    base = float(np.clip(np.mean(np.asarray(y)), 1e-4, 1 - 1e-4))
    params = {
        "w": jnp.zeros(x.shape[1], dtype=jnp.float32),
        "b": jnp.asarray(np.log(base / (1 - base)), dtype=jnp.float32),
    }
    optimiser = optax.adam(learning_rate)
    state = optimiser.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(_loss_fn)(params, x, y, l2)
        updates, state = optimiser.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    loss = jnp.asarray(0.0)
    for _ in range(steps):
        params, state, loss = step(params, state)
    return params, float(loss)


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l2: float,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    standardizer: Standardizer | None = None,
) -> LinearProbe:
    """Fit one logistic probe at one regularisation strength."""
    standardizer = standardizer or Standardizer.fit(x_train)
    z = standardizer.apply(x_train)
    params, loss = _fit(z, np.asarray(y_train), l2, steps, learning_rate)
    return LinearProbe(
        weights=np.asarray(params["w"], dtype=np.float32),
        bias=float(params["b"]),
        l2=l2,
        standardizer=standardizer,
        final_loss=loss,
    )


def select_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    l2_grid=L2_GRID,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    score_fn=None,
) -> tuple[LinearProbe, list[dict]]:
    """Fit the grid on train, keep the strength that scores best on validation.

    `score_fn(scores, labels) -> float` defaults to AUROC, the metric the
    results are reported in. The test split is not involved.
    """
    if score_fn is None:
        from answer_probe.metrics import auroc as score_fn  # noqa: N813

    standardizer = Standardizer.fit(x_train)
    trace: list[dict] = []
    best: LinearProbe | None = None
    best_score = -np.inf
    for l2 in l2_grid:
        probe = fit_linear_probe(
            x_train,
            y_train,
            l2=l2,
            steps=steps,
            learning_rate=learning_rate,
            standardizer=standardizer,
        )
        val_score = float(score_fn(probe.score(x_val), y_val))
        trace.append({"l2": l2, "val_score": val_score, "train_loss": probe.final_loss})
        if val_score > best_score:
            best, best_score = probe, val_score
    assert best is not None
    return best, trace


@dataclass(frozen=True)
class DiffOfMeansProbe:
    """Project onto the difference between the wrong-answer and right-answer means.

    No fitting beyond those two means. The score is unnormalised, which is fine
    for ranking metrics; it is deliberately not turned into a probability,
    because doing so would mean training something.
    """

    direction: np.ndarray
    standardizer: Standardizer

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, standardizer: Standardizer | None = None):
        standardizer = standardizer or Standardizer.fit(x)
        z = standardizer.apply(x)
        y = np.asarray(y).astype(bool)
        if not y.any() or not (~y).any():
            raise ValueError("need both right and wrong answers to form a direction")
        direction = z[y].mean(axis=0) - z[~y].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return cls(direction.astype(np.float32), standardizer)

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.standardizer.apply(x) @ self.direction, dtype=np.float64)


def shuffle_labels(y: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute labels, for the control that must land near chance."""
    rng = np.random.default_rng(seed)
    return np.asarray(y)[rng.permutation(len(y))]
