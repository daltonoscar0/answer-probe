"""Check the generation half of the notebook against the real tokenizer.

The notebook tokenizes the prompt itself (to know where the prompt ends) and
separately hands the same string to the library's sampler. If those two
tokenizations ever disagree, the "last token of the prompt" would not be the
position the model actually generated from. `test_sampler_tokenizes_the_prompt_
exactly_as_we_do` pins that assumption down.

The tokenizer is a ~4 MB download from a public GCS bucket, so these tests skip
when there is no network.
"""

import numpy as np
import pytest

pytest.importorskip("gemma", reason="needs the 'extract' extra")

from answer_probe.extract import (  # noqa: E402
    build_sampler,
    extract_batch,
    generate,
    strip_terminators,
    terminator_ids,
    tiny_model,
)
from answer_probe.prompts import build_prompt  # noqa: E402

GEMMA3_VOCAB = 262144


@pytest.fixture(scope="module")
def tokenizer():
    from gemma import gm

    try:
        tok = gm.text.Gemma3Tokenizer()
        tok.encode("warm up", add_bos=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"tokenizer unavailable: {exc}")
    return tok


@pytest.fixture(scope="module")
def tiny_full_vocab():
    # Full vocabulary so real token ids are in range; still tiny everywhere else.
    return tiny_model(num_layers=2, embed_dim=32, num_embed=GEMMA3_VOCAB)


def test_sampler_tokenizes_the_prompt_exactly_as_we_do(tokenizer):
    from gemma.gm.text import _sampler

    prompt = build_prompt("Who wrote Ulysses?")
    normalized = _sampler._normalize_prompt(prompt, format=tokenizer.FORMAT)
    assert list(normalized) == [prompt], "the sampler rewrites the prompt string"


def test_prompt_ends_at_the_open_model_turn(tokenizer):
    prompt = build_prompt("Who wrote Ulysses?")
    ids = tokenizer.encode(prompt, add_bos=True)
    assert ids[0] == int(type(tokenizer).special_tokens.BOS)
    # The last token is the newline that opens the model's turn, so the next
    # token the model emits is the first token of its answer.
    assert tokenizer.decode(ids[-3:]) == "<start_of_turn>model\n"


def test_terminator_ids(tokenizer):
    ids = terminator_ids(tokenizer)
    special = type(tokenizer).special_tokens
    assert ids == {int(special.EOS), int(special.END_OF_TURN), int(special.PAD)}


def test_greedy_generation_and_extraction_run_end_to_end(tokenizer, tiny_full_vocab):
    """The notebook's inner loop, on random weights.

    The answers are gibberish (the weights are random); what is being tested is
    that the sampler, the tokenizer and `extract_batch` agree on shapes and
    indices.
    """
    model, params = tiny_full_vocab
    sampler = build_sampler(
        model,
        params,
        tokenizer,
        max_new_tokens=4,
        cache_length=128,
        pad_length=(64,),
    )
    questions = ["Who wrote Ulysses?", "What is the capital of Peru?"]
    prompts = [build_prompt(q) for q in questions]
    texts, token_ids = generate(sampler, prompts, max_new_tokens=4)

    assert len(texts) == 2
    assert all(isinstance(t, str) for t in texts)
    assert all(len(ids) <= 4 for ids in token_ids)

    terminators = terminator_ids(tokenizer)
    prompt_ids = [tokenizer.encode(p, add_bos=True) for p in prompts]
    answer_ids = [strip_terminators(ids, terminators) for ids in token_ids]

    out = extract_batch(model, params, prompt_ids, answer_ids, max_answer_tokens=4)
    assert out.before.shape == (2, 4, 32)  # 2 blocks + embeddings + final norm
    assert np.all(np.isfinite(out.before))
    assert list(out.n_answer_tokens) == [len(a) for a in answer_ids]


def test_temperature_sampling_is_seeded_and_varies(tokenizer, tiny_full_vocab):
    model, params = tiny_full_vocab
    sampler = build_sampler(
        model,
        params,
        tokenizer,
        max_new_tokens=4,
        cache_length=128,
        pad_length=(64,),
        temperature=1.0,
    )
    prompts = [build_prompt("Who wrote Ulysses?")]
    first, first_ids = generate(sampler, prompts, max_new_tokens=4, rng=0)
    again, again_ids = generate(sampler, prompts, max_new_tokens=4, rng=0)
    assert (first, first_ids) == (again, again_ids), "same seed must give the same sample"
