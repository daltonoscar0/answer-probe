"""Exercise the real extraction code on a tiny randomly initialised Gemma.

Same block implementation and same `extract_batch` the GPU run uses, at a size
that fits on a CPU in CI. The two checks that matter scientifically are
`test_before_position_is_not_contaminated_by_the_answer` (the headline position
really is the state before any answer token exists) and
`test_padding_does_not_change_results`.
"""

import numpy as np
import pytest

gemma = pytest.importorskip("gemma", reason="needs the 'extract' extra")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from answer_probe.extract import (  # noqa: E402
    FP16_MAX,
    PAD_ID,
    extract_batch,
    n_layers_recorded,
    strip_terminators,
    to_float16,
    tiny_model,
)

NUM_LAYERS = 4
EMBED_DIM = 32
VOCAB = 128


@pytest.fixture(scope="module")
def tiny():
    return tiny_model(num_layers=NUM_LAYERS, embed_dim=EMBED_DIM, num_embed=VOCAB)


def _prompt(length, start=10):
    return [2] + list(range(start, start + length - 1))


def test_shapes(tiny):
    model, params = tiny
    prompts = [_prompt(12), _prompt(9, start=40)]
    answers = [[31, 32, 33], [44]]
    out = extract_batch(model, params, prompts, answers, max_answer_tokens=8)

    n_layers = n_layers_recorded(model)
    assert n_layers == NUM_LAYERS + 2
    assert out.before.shape == (2, n_layers, EMBED_DIM)
    assert out.after.shape == (2, n_layers, EMBED_DIM)
    assert out.first_token_prob.shape == (2,)
    assert list(out.n_answer_tokens) == [3, 1]
    assert np.all(np.isfinite(out.before)) and np.all(np.isfinite(out.after))


def test_confidence_signals_are_in_range(tiny):
    model, params = tiny
    out = extract_batch(
        model, params, [_prompt(10)], [[31, 32]], max_answer_tokens=8
    )
    assert 0.0 < out.first_token_prob[0] <= 1.0
    assert out.first_token_entropy[0] >= 0.0
    # Uniform over `VOCAB` tokens is the most uncertain the model can be.
    assert out.first_token_entropy[0] <= np.log(VOCAB) + 1e-4
    assert out.answer_logprob_mean[0] <= 0.0
    assert out.answer_logprob_sum[0] == pytest.approx(
        out.answer_logprob_mean[0] * 2, rel=1e-5
    )


def test_last_recorded_layer_matches_the_library(tiny):
    """Layer N+1 must be the model's own final hidden state, not something else."""
    model, params = tiny
    tokens = jnp.asarray([_prompt(12)], dtype=jnp.int32)
    out, mutated = model.apply(
        {"params": params},
        tokens=tokens,
        return_hidden_states=True,
        mutable=["intermediates"],
    )
    residual = mutated["intermediates"]["residual"]
    assert len(residual) == NUM_LAYERS + 2
    np.testing.assert_allclose(
        np.asarray(residual[-1]), np.asarray(out.hidden_states), rtol=1e-6, atol=1e-6
    )


def test_before_position_is_not_contaminated_by_the_answer(tiny):
    """The headline position is the state before any answer token exists.

    Attention is causal, so running prompt+answer in one pass and reading the
    last prompt token must give exactly what a prompt-only pass gives. If this
    ever fails, the "before answering" claim is false.
    """
    model, params = tiny
    prompt = _prompt(11)
    with_answer = extract_batch(
        model, params, [prompt], [[31, 32, 33, 34]], max_answer_tokens=8
    )
    prompt_only = extract_batch(model, params, [prompt], [[]], max_answer_tokens=8)
    np.testing.assert_allclose(
        with_answer.before, prompt_only.before, rtol=2e-5, atol=2e-5
    )
    # With no answer tokens, "after" degenerates to the prompt position.
    np.testing.assert_allclose(
        prompt_only.before, prompt_only.after, rtol=1e-6, atol=1e-6
    )
    assert prompt_only.n_answer_tokens[0] == 0


def test_padding_does_not_change_results(tiny):
    """A short question batched with a long one must score as it does alone."""
    model, params = tiny
    prompts = [_prompt(8), _prompt(30, start=50)]
    answers = [[31, 32], [41, 42, 43, 44, 45]]

    batched = extract_batch(model, params, prompts, answers, max_answer_tokens=8)
    for i in range(2):
        alone = extract_batch(
            model, params, [prompts[i]], [answers[i]], max_answer_tokens=8
        )
        np.testing.assert_allclose(batched.before[i], alone.before[0], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(batched.after[i], alone.after[0], rtol=2e-4, atol=2e-4)
        assert batched.first_token_prob[i] == pytest.approx(
            alone.first_token_prob[0], rel=2e-4
        )
        assert batched.answer_logprob_mean[i] == pytest.approx(
            alone.answer_logprob_mean[0], rel=2e-4
        )


def test_first_token_prob_matches_a_hand_computed_softmax(tiny):
    model, params = tiny
    prompt = _prompt(10)
    first_answer_token = 37
    out = extract_batch(
        model, params, [prompt], [[first_answer_token, 38]], max_answer_tokens=8
    )

    logits = model.apply({"params": params}, tokens=jnp.asarray([prompt])).logits
    probs = jax.nn.softmax(logits[0, -1].astype(jnp.float32))
    expected_entropy = -np.sum(np.asarray(probs) * np.log(np.asarray(probs) + 1e-30))

    assert out.first_token_prob[0] == pytest.approx(float(probs[first_answer_token]), rel=1e-4)
    assert out.first_token_entropy[0] == pytest.approx(expected_entropy, rel=1e-3)


def test_strip_terminators():
    assert strip_terminators([5, 6, 106, 1], {1, 106, PAD_ID}) == [5, 6]
    assert strip_terminators([5, 6], {1, 106}) == [5, 6]
    assert strip_terminators([106], {1, 106}) == []
    assert strip_terminators([], {1}) == []


def test_to_float16_clips_instead_of_overflowing():
    array = np.array([[1.0, 1e6, -1e6, np.inf, np.nan]], dtype=np.float32)
    out, n_clipped = to_float16(array)
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(FP16_MAX)
    assert out[0, 2] == pytest.approx(-FP16_MAX)
    assert n_clipped == 2  # the two 1e6 values; inf/nan are mapped first


def test_to_float16_leaves_ordinary_values_alone():
    rng = np.random.default_rng(0)
    array = rng.normal(size=(4, 8)).astype(np.float32)
    out, n_clipped = to_float16(array)
    assert n_clipped == 0
    np.testing.assert_allclose(out.astype(np.float32), array, rtol=1e-3, atol=1e-3)
