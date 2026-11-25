"""Generation and residual-stream extraction, in JAX.

Run on a GPU from `notebooks/extract.ipynb`. Nothing in the analysis pipeline
imports this module, so `pip install -e .` stays light; the `extract` extra
pulls in the DeepMind `gemma` library.

Two things happen per question:

1. The model answers greedily (the library's KV-cached sampler).
2. One teacher-forced forward pass over prompt + answer records the residual
   stream at every layer and every confidence signal we need. Because attention
   is causal, the state at the last *prompt* token in that pass is bit-identical
   to the state the model had before it produced any answer token, so the
   headline "before answering" position costs no extra compute.

Layer indexing. Layer 0 is the embedding output, layers 1..N are the outputs of
the N transformer blocks, and layer N+1 is the final RMSNorm output: N+2 rows in
total for an N-layer model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


def patch_numpy_long_double() -> None:
    """Make `kauldron.ktyping` importable on platforms without 128-bit floats.

    kauldron registers `np.float128`/`np.complex256`, which exist on x86 Linux
    (Colab, CI) but not on Apple Silicon. Called before the gemma import so the
    tiny-model test can also run on a Mac.
    """
    for name, fallback in (("float128", np.longdouble), ("complex256", np.clongdouble)):
        if not hasattr(np, name):
            setattr(np, name, fallback)


patch_numpy_long_double()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Sentinel for "no token here". Gemma's PAD id is 0 and no real token is 0.
PAD_ID = 0

# Sequence lengths are rounded up to one of these before the forward pass, so
# XLA compiles a handful of shapes instead of one per batch.
LENGTH_BUCKETS = (64, 96, 128, 192, 256)


def residual_model_class(base_cls: type) -> type:
    """Subclass a gemma model so `apply(..., mutable=['intermediates'])` records
    the residual stream after every block.

    The library's own `return_hidden_states=True` gives only the final layer.
    This override reimplements the block loop verbatim and `sow`s `x` at each
    step; when 'intermediates' is not mutable (the sampler's own calls) `sow` is
    a no-op, so the same model object can be used for generation.
    """

    class _WithResiduals(base_cls):  # type: ignore[misc, valid-type]
        def _apply_attention(self, inputs, cache):
            x = inputs.embeddings
            self.sow("intermediates", "residual", x)
            old_cache = cache or {}
            new_cache = {}
            for i, block in enumerate(self.blocks):
                layer_name = f"layer_{i}"
                layer_cache, x = block(
                    x,
                    inputs.positions,
                    old_cache.get(layer_name),
                    inputs.attention_mask,
                )
                new_cache[layer_name] = layer_cache
                self.sow("intermediates", "residual", x)
            x = self.final_norm(x)
            self.sow("intermediates", "residual", x)
            return x, new_cache

    _WithResiduals.__name__ = f"{base_cls.__name__}WithResiduals"
    _WithResiduals.__qualname__ = _WithResiduals.__name__
    return _WithResiduals


def n_layers_recorded(model: Any) -> int:
    """How many rows the layer axis of the saved arrays will have."""
    return model.config.num_layers + 2


@functools.partial(jax.jit, static_argnames=("model", "n_answer_slots"))
def _forward(model, params, tokens, prompt_last, answer_last, targets, n_answer_slots):
    """One teacher-forced pass. Returns only small arrays.

    Args:
      tokens: `[B, L]` padded prompt+answer token ids.
      prompt_last: `[B]` index of the last prompt token.
      answer_last: `[B]` index of the last answer content token.
      targets: `[B, n_answer_slots]` answer token ids, PAD-filled.
    """
    out, mutated = model.apply(
        {"params": params},
        tokens=tokens,
        mutable=["intermediates"],
    )
    residual = jnp.stack(mutated["intermediates"]["residual"], axis=1)  # [B, NL, L, D]

    def gather(index):
        idx = index[:, None, None, None]
        return jnp.take_along_axis(residual, idx, axis=2)[:, :, 0, :]  # [B, NL, D]

    before = gather(prompt_last)
    after = gather(answer_last)

    # Logits at position t predict the token at position t+1, so the slots that
    # predict the answer start at the last prompt token.
    offsets = prompt_last[:, None] + jnp.arange(n_answer_slots)[None, :]
    offsets = jnp.clip(offsets, 0, tokens.shape[1] - 1)
    answer_logits = jnp.take_along_axis(
        out.logits, offsets[:, :, None], axis=1
    ).astype(jnp.float32)  # [B, n_answer_slots, V]

    logprobs = jax.nn.log_softmax(answer_logits, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, targets[:, :, None], axis=-1)[:, :, 0]

    first = logprobs[:, 0, :]
    first_token_logprob = token_logprobs[:, 0]
    entropy = -jnp.sum(jnp.exp(first) * first, axis=-1)

    valid = (targets != PAD_ID).astype(jnp.float32)
    n_valid = jnp.maximum(valid.sum(axis=-1), 1.0)
    mean_logprob = (token_logprobs * valid).sum(axis=-1) / n_valid
    sum_logprob = (token_logprobs * valid).sum(axis=-1)

    return {
        "before": before,
        "after": after,
        "first_token_logprob": first_token_logprob,
        "first_token_entropy": entropy,
        "answer_logprob_mean": mean_logprob,
        "answer_logprob_sum": sum_logprob,
    }


def bucket_length(length: int) -> int:
    for bucket in LENGTH_BUCKETS:
        if length <= bucket:
            return bucket
    return length


@dataclass
class BatchSignals:
    """Per-question outputs of one extraction batch."""

    before: np.ndarray  # [B, n_layers, D] float32
    after: np.ndarray  # [B, n_layers, D] float32
    first_token_prob: np.ndarray  # [B]
    first_token_entropy: np.ndarray  # [B]
    answer_logprob_mean: np.ndarray  # [B]
    answer_logprob_sum: np.ndarray  # [B]
    n_answer_tokens: np.ndarray  # [B] int


def strip_terminators(answer_ids: list[int], terminators: set[int]) -> list[int]:
    """Drop the trailing end-of-turn/EOS/padding from a generated answer.

    Only the content tokens are scored and only a content token is used as the
    "after answering" position, so the two positions are comparable across
    answers of different lengths.
    """
    out = list(answer_ids)
    while out and out[-1] in terminators:
        out.pop()
    return out


def extract_batch(
    model,
    params,
    prompt_ids: list[list[int]],
    answer_ids: list[list[int]],
    *,
    max_answer_tokens: int,
) -> BatchSignals:
    """Residual states at both positions, plus confidence signals, for a batch.

    `answer_ids` must already have terminators stripped (see
    `strip_terminators`).
    """
    if len(prompt_ids) != len(answer_ids):
        raise ValueError("prompt_ids and answer_ids must be the same length")
    batch = len(prompt_ids)
    lengths = [len(p) + len(a) for p, a in zip(prompt_ids, answer_ids, strict=True)]
    width = bucket_length(max(lengths))

    tokens = np.full((batch, width), PAD_ID, dtype=np.int32)
    targets = np.full((batch, max_answer_tokens), PAD_ID, dtype=np.int32)
    prompt_last = np.zeros(batch, dtype=np.int32)
    answer_last = np.zeros(batch, dtype=np.int32)
    n_answer = np.zeros(batch, dtype=np.int32)

    for i, (prompt, answer) in enumerate(zip(prompt_ids, answer_ids, strict=True)):
        answer = answer[:max_answer_tokens]
        seq = list(prompt) + list(answer)
        tokens[i, : len(seq)] = seq
        prompt_last[i] = len(prompt) - 1
        # With no answer tokens there is no "after answering" position; fall
        # back to the prompt position and record the zero length so the
        # analysis can exclude the question.
        answer_last[i] = len(seq) - 1 if answer else len(prompt) - 1
        targets[i, : len(answer)] = answer
        n_answer[i] = len(answer)

    out = _forward(
        model,
        params,
        jnp.asarray(tokens),
        jnp.asarray(prompt_last),
        jnp.asarray(answer_last),
        jnp.asarray(targets),
        max_answer_tokens,
    )
    out = jax.tree.map(np.asarray, out)
    return BatchSignals(
        before=out["before"].astype(np.float32),
        after=out["after"].astype(np.float32),
        first_token_prob=np.exp(out["first_token_logprob"]),
        first_token_entropy=out["first_token_entropy"],
        answer_logprob_mean=out["answer_logprob_mean"],
        answer_logprob_sum=out["answer_logprob_sum"],
        n_answer_tokens=n_answer,
    )


# bfloat16 is what the activation files are stored as: two bytes per value, the
# same as float16, but with float32's exponent range. Gemma's residual stream
# has large outliers -- measured at 2.2e4 on the 270M model, against a float16
# ceiling of 6.55e4 -- and bfloat16's ceiling of 3.4e38 puts them nowhere near
# it, so nothing is clipped. It is also the dtype the model computes in, so
# saving the states costs no precision at all: the float16 cast this replaced
# both clipped and re-rounded.
BF16 = ml_dtypes.bfloat16
BF16_MAX = float(np.finfo(BF16).max)


def to_bfloat16(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Cast to bfloat16. Returns (array, n_out_of_range), the latter always 0.

    The count is kept and recorded so the claim "nothing was clipped" is a
    measurement rather than an assumption; `run_greedy_pass` asserts it.
    Non-finite values are a real failure and raise rather than being quietly
    mapped to something finite.
    """
    array = np.asarray(array)
    if not np.isfinite(array).all():
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{n_bad} non-finite values in the residual stream; the forward pass "
            "produced inf or nan and the run cannot be trusted"
        )
    n_out_of_range = int(np.sum(np.abs(array) > BF16_MAX))
    return array.astype(BF16), n_out_of_range


# The instruction-tuned Gemma 3 text models that fit a T4 in bfloat16, with the
# public checkpoint each one loads from. `n_blocks` is recorded so a run can be
# checked against the architecture it claims to come from; the saved arrays have
# `n_blocks + 2` layers (embeddings and the final norm).
CHECKPOINTS: dict[str, dict] = {
    "gemma3-270m-it": {
        "class": "Gemma3_270M",
        "path": "gs://gemma-data/checkpoints/gemma3-270m-it",
        "n_blocks": 18,
        "hidden_size": 640,
    },
    "gemma3-1b-it": {
        "class": "Gemma3_1B",
        "path": "gs://gemma-data/checkpoints/gemma3-1b-it",
        "n_blocks": 26,
        "hidden_size": 1152,
    },
    "gemma3-4b-it": {
        "class": "Gemma3_4B",
        "path": "gs://gemma-data/checkpoints/gemma3-4b-it",
        "n_blocks": 34,
        "hidden_size": 2560,
    },
}


def load_model(name: str, *, dtype: str = "bfloat16", cache_dir="checkpoints"):
    """Build the residual-recording model, restore its weights, and tokenize.

    Returns `(model, params, tokenizer)`.
    """
    from gemma import gm

    if name not in CHECKPOINTS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(CHECKPOINTS)}")
    spec = CHECKPOINTS[name]
    local = download_checkpoint(spec["path"], Path(cache_dir) / name)
    base_cls = getattr(gm.nn, spec["class"])
    model = residual_model_class(base_cls)(dtype=getattr(jnp, dtype))
    params = gm.ckpts.load_params(str(local), text_only=True)
    tokenizer = gm.text.Tokenizer.from_version(base_cls.INFO.tokenizer_version)

    if model.config.num_layers != spec["n_blocks"]:
        raise RuntimeError(
            f"{name} has {model.config.num_layers} blocks, expected {spec['n_blocks']}"
        )
    return model, params, tokenizer


GEMMA_BUCKET = "gemma-data"
_GCS_LIST = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
_GCS_GET = "https://storage.googleapis.com/{bucket}/{name}"


def download_checkpoint(checkpoint_path: str, dest, *, progress: bool = True) -> Path:
    """Copy a `gs://gemma-data/...` checkpoint to local disk over plain HTTPS.

    The bucket is public, so no credentials are needed and no Hugging Face
    token is involved: the JAX `gemma` library distributes weights here, not on
    the Hub. Orbax can read `gs://` paths directly, but it goes through
    google-cloud-storage, which spends a while failing to find credentials
    first. Fetching the objects over plain HTTPS is simpler, and because files
    already present with the right size are skipped, the destination can live
    on Drive and survive a disconnected Colab session.
    """
    import urllib.parse
    import urllib.request

    if not checkpoint_path.startswith("gs://"):
        return Path(checkpoint_path).resolve()
    bucket, _, prefix = checkpoint_path[len("gs://") :].partition("/")
    prefix = prefix.rstrip("/") + "/"
    # Absolute: orbax rejects relative checkpoint paths, and the error it gives
    # for a relative one is reported as missing metadata rather than a bad path.
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    objects, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name,size),nextPageToken"}
        if token:
            query["pageToken"] = token
        url = _GCS_LIST.format(bucket=bucket) + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url) as response:  # noqa: S310
            import json as _json

            payload = _json.load(response)
        objects.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            break
    if not objects:
        raise RuntimeError(f"no objects under {checkpoint_path}")

    total = sum(int(o["size"]) for o in objects)
    done = 0
    for obj in objects:
        name = obj["name"]
        size = int(obj["size"])
        # Orbax writes placeholder "<dir>_$folder$" markers; they are not files.
        if name.endswith("/") or name.endswith("_$folder$"):
            done += size
            continue
        target = dest / name[len(prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == size:
            done += size
            continue
        url = _GCS_GET.format(bucket=bucket, name=urllib.parse.quote(name))
        urllib.request.urlretrieve(url, target)  # noqa: S310
        done += size
        if progress:
            print(f"  {done / 1e6:7.0f} / {total / 1e6:.0f} MB  {target.name}", flush=True)
    return dest


def terminator_ids(tokenizer) -> set[int]:
    """Token ids that end a model turn and are not part of the answer text."""
    special = type(tokenizer).special_tokens
    return {int(special.EOS), int(special.END_OF_TURN), int(special.PAD)}


def build_sampler(
    model,
    params,
    tokenizer,
    *,
    max_new_tokens: int,
    cache_length: int,
    pad_length: tuple[int, ...],
    temperature: float | None = None,
):
    """A greedy sampler, or a temperature sampler when `temperature` is given.

    `cache_length` and `pad_length` are kept small on purpose: the library
    defaults (4096 / 2048) allocate a KV cache far larger than short-answer
    prompts need, which on a 16 GB T4 is the difference between running and
    running out of memory.
    """
    from gemma import gm

    sampling = (
        gm.text.Greedy() if temperature is None else gm.text.RandomSampling(temperature=temperature)
    )
    return gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        sampling=sampling,
        cache_length=cache_length,
        max_out_length=max_new_tokens,
        pad_length=pad_length,
    )


def generate(
    sampler,
    prompts: list[str],
    *,
    max_new_tokens: int,
    terminators: set[int],
    rng=None,
) -> tuple[list[str], list[list[int]]]:
    """Answer a batch of prompts. Returns (answer texts, answer token ids).

    The text is decoded from the same token ids that are returned, after the
    end-of-turn marker is stripped, so the saved answer string and the tokens
    the forward pass scores are always the same thing. The sampler's own
    `.text` keeps the literal `<end_of_turn>` in the string.
    """
    out = sampler.sample(
        prompts,
        max_new_tokens=max_new_tokens,
        return_state=True,
        **({} if rng is None else {"rng": rng}),
    )
    tokens = np.asarray(out.tokens)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    ids = [
        strip_terminators([int(t) for t in row if int(t) != PAD_ID], terminators)
        for row in tokens
    ]
    texts = [sampler.tokenizer.decode(row).strip() if row else "" for row in ids]
    return texts, ids


def tiny_model(*, num_layers: int = 4, embed_dim: int = 32, num_embed: int = 128):
    """A randomly initialised Gemma 3 of toy size, for tests without a GPU.

    Same code path as the real model: same block implementation, same attention
    pattern, same residual recording.
    """
    from gemma.gm.nn import _config, _gemma, _transformer

    config = _config.TransformerConfig(
        final_logit_softcap=None,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=2 * embed_dim,
        num_heads=2,
        head_dim=16,
        num_kv_heads=1,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=_config.make_attention_layers_types(
            _gemma.GEMMA3_ATTENTION_PATTERN, num_layers=num_layers
        ),
        query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=16,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        vision_encoder=None,
    )

    cls = residual_model_class(_transformer.Transformer)
    model = cls(config=config, dtype=jnp.float32)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), tokens=tokens)["params"]
    return model, params
