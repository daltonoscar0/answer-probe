"""The notebook's two passes, run for real on a tiny model, then analysed.

This is the closest thing to running `notebooks/extract.ipynb` that fits in CI:
the same functions the notebook calls, the same tokenizer, the same
checkpointing, the same run-folder writer, and then the same four CLI commands.
A tiny randomly initialised Gemma answers nonsense, so no number here means
anything -- but if the notebook would crash, or write a folder the analysis
cannot read, this test fails.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gemma", reason="needs the 'extract' extra")

import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from answer_probe import cli, io  # noqa: E402
from answer_probe.extract import (  # noqa: E402
    build_sampler,
    n_layers_recorded,
    terminator_ids,
    tiny_model,
)
from answer_probe.pipeline import run_greedy_pass, run_selfconsistency_pass  # noqa: E402
from answer_probe.splits import make_splits, selfconsistency_qids  # noqa: E402

GEMMA3_VOCAB = 262144
MAX_NEW_TOKENS = 4
N_MAIN, N_SHIFT = 24, 10


@pytest.fixture(scope="module")
def tokenizer():
    from gemma import gm

    try:
        tok = gm.text.Gemma3Tokenizer()
        tok.encode("warm up", add_bos=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"tokenizer unavailable: {exc}")
    return tok


def make_questions():
    questions = []
    for dataset, count in (("triviaqa", N_MAIN), ("nq_open", N_SHIFT)):
        for i in range(count):
            questions.append(
                {
                    "qid": f"{dataset}:{i}",
                    "dataset": dataset,
                    "question": f"Who wrote book number {i}?",
                    "aliases": ["Paris", "1969"],
                }
            )
    return questions


@pytest.fixture(scope="module")
def gpu_path(tmp_path_factory, tokenizer):
    """Run both passes and assemble the run folder, exactly as the notebook does."""
    model, params = tiny_model(num_layers=2, embed_dim=32, num_embed=GEMMA3_VOCAB)
    terminators = terminator_ids(tokenizer)
    questions = make_questions()
    root = tmp_path_factory.mktemp("gpu_path")

    greedy_sampler = build_sampler(
        model, params, tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, cache_length=128, pad_length=(64,),
    )
    greedy = run_greedy_pass(
        model, params, greedy_sampler, tokenizer, questions,
        shard_dir=root / "ckpt" / "greedy",
        shard_size=10, gen_batch=8, extract_batch_size=4,
        max_new_tokens=MAX_NEW_TOKENS, terminators=terminators,
        log=lambda *_: None,
    )

    splits = make_splits(
        [q["qid"] for q in questions], [q["dataset"] for q in questions]
    )
    wanted = set(selfconsistency_qids(splits))
    targets = [q for q in questions if q["qid"] in wanted]
    sample_sampler = build_sampler(
        model, params, tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, cache_length=128, pad_length=(64,),
        temperature=1.0,
    )
    selfcons = run_selfconsistency_pass(
        sample_sampler, targets,
        {g["qid"]: g["answer"] for g in greedy.generations},
        shard_dir=root / "ckpt" / "selfcons",
        shard_size=10, gen_batch=8, max_new_tokens=MAX_NEW_TOKENS,
        terminators=terminators, n_samples=3, seed=11,
        log=lambda *_: None,
    )

    # A tiny random model answers gibberish, so every answer would be scored
    # wrong and the probes would have one class to separate from nothing. Give
    # half the questions the answer the model actually produced, so the
    # analysis path runs on a realistic mix.
    answers = {g["qid"]: g["answer"] for g in greedy.generations}
    for i, question in enumerate(questions):
        if i % 2 == 0 and answers[question["qid"]].strip():
            question["aliases"] = [answers[question["qid"]]]

    run_dir = root / "run"
    io.write_run_folder(
        run_dir,
        manifest={"model": "tiny-random", "checkpoint": "none", "model_dtype": "float32"},
        questions=questions,
        generations=greedy.generations,
        selfconsistency=selfcons,
        before=greedy.before,
        after=greedy.after,
        qids=[q["qid"] for q in questions],
        data_summary=pd.DataFrame([{"dataset": "triviaqa", "n_sampled": N_MAIN}]),
        split_map=splits,
    )
    return {
        "root": root,
        "run_dir": run_dir,
        "greedy": greedy,
        "selfcons": selfcons,
        "questions": questions,
        "splits": splits,
        "model": model,
    }


def test_greedy_pass_answers_every_question_in_order(gpu_path):
    greedy = gpu_path["greedy"]
    expected = [q["qid"] for q in gpu_path["questions"]]
    assert [g["qid"] for g in greedy.generations] == expected
    assert greedy.before.shape == (
        len(expected),
        n_layers_recorded(gpu_path["model"]),
        32,
    )
    assert greedy.before.dtype == jnp.float16
    for record in greedy.generations:
        assert 0.0 < record["first_token_prob"] <= 1.0
        assert record["first_token_entropy"] >= 0.0
        assert record["answer_logprob_mean"] <= 0.0
        assert "<end_of_turn>" not in record["answer"]


def test_greedy_pass_resumes_instead_of_regenerating(gpu_path, tokenizer):
    """Re-running over existing checkpoints must change nothing."""
    from answer_probe.extract import tiny_model as _tiny

    model, params = gpu_path["model"], None
    del model, params
    model, params = _tiny(num_layers=2, embed_dim=32, num_embed=GEMMA3_VOCAB)
    sampler = build_sampler(
        model, params, tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, cache_length=128, pad_length=(64,),
    )
    again = run_greedy_pass(
        model, params, sampler, tokenizer, gpu_path["questions"],
        shard_dir=gpu_path["root"] / "ckpt" / "greedy",
        shard_size=10, gen_batch=8, extract_batch_size=4,
        max_new_tokens=MAX_NEW_TOKENS, terminators=terminator_ids(tokenizer),
        log=lambda *_: None,
    )
    assert again.elapsed_seconds < gpu_path["greedy"].elapsed_seconds
    assert [g["qid"] for g in again.generations] == [
        g["qid"] for g in gpu_path["greedy"].generations
    ]
    assert again.generations == gpu_path["greedy"].generations


def test_greedy_pass_refuses_a_shrunken_question_set(gpu_path, tokenizer):
    from answer_probe.extract import tiny_model as _tiny

    model, params = _tiny(num_layers=2, embed_dim=32, num_embed=GEMMA3_VOCAB)
    sampler = build_sampler(
        model, params, tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, cache_length=128, pad_length=(64,),
    )
    with pytest.raises(RuntimeError, match="sampled"):
        run_greedy_pass(
            model, params, sampler, tokenizer, gpu_path["questions"][:4],
            shard_dir=gpu_path["root"] / "ckpt" / "greedy",
            shard_size=10, gen_batch=8, extract_batch_size=4,
            max_new_tokens=MAX_NEW_TOKENS, terminators=terminator_ids(tokenizer),
            log=lambda *_: None,
        )


def test_selfconsistency_covers_the_evaluation_splits_only(gpu_path):
    wanted = set(selfconsistency_qids(gpu_path["splits"]))
    covered = {r["qid"] for r in gpu_path["selfcons"]}
    assert covered == wanted
    assert not covered & set(gpu_path["splits"]["train"])
    for record in gpu_path["selfcons"]:
        assert 0.0 <= record["agreement"] <= 1.0
        assert len(record["samples"]) == record["n_samples"] == 3


def test_the_analysis_cli_reads_the_folder_the_notebook_writes(gpu_path):
    """The handover point: whatever the notebook writes, `aprobe` must consume."""
    root = gpu_path["root"]
    io.set_dirs(data=root / "data", results=root / "results")
    try:
        runner = CliRunner()
        for args in (
            ["label", "--run-dir", str(gpu_path["run_dir"])],
            ["train", "--run-dir", str(gpu_path["run_dir"])],
        ):
            result = runner.invoke(cli.app, args)
            assert result.exit_code == 0, (
                f"`aprobe {' '.join(args)}` failed:\n{result.output}\n{result.exception!r}"
            )
        labels = pd.read_csv(root / "results" / "labels.csv")
        assert len(labels) == N_MAIN + N_SHIFT
        assert set(labels["correct"].unique()) == {0, 1}, "fixture needs both classes"
        assert (root / "results" / "scores.csv").exists()
        scan = pd.read_csv(root / "results" / "layer_scan.csv")
        assert sorted(scan["layer"].unique()) == list(
            range(n_layers_recorded(gpu_path["model"]))
        )
    finally:
        io.set_dirs(data=io.REPO_ROOT / "data", results=io.REPO_ROOT / "results")
