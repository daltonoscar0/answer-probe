"""The two GPU passes the notebook runs.

They live here rather than in notebook cells so they can be tested: the
end-to-end test in `tests/test_pipeline_gpu_path.py` runs both of them on a
tiny randomly initialised Gemma and feeds the result through the analysis CLI.

The greedy pass must finish and be checkpointed before the self-consistency
pass begins. Self-consistency costs eight generations per question, and a
session that runs out of time partway through it should still leave the core
data complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from answer_probe.extract import extract_batch, generate, to_float16
from answer_probe.normalize import is_correct, normalize_answer
from answer_probe.prompts import build_prompt
from answer_probe.shards import Progress, ShardWriter, format_duration

EARLY_CHECK_AFTER = 200
USABLE_ACCURACY = (0.10, 0.90)


@dataclass
class GreedyResult:
    """Everything the greedy pass produced, read back from its checkpoints."""

    generations: list[dict]
    before: np.ndarray
    after: np.ndarray
    max_abs_residual: float = 0.0
    values_clipped: int = 0
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def _early_accuracy_note(correct: list[bool], progress: Progress) -> list[str]:
    accuracy = float(np.mean(correct))
    notes = [
        f"After {len(correct)} main-set questions: {accuracy:.1%} correct.",
        "Projected time for the whole greedy pass: "
        f"{format_duration(progress.elapsed() + progress.remaining())}.",
    ]
    if not USABLE_ACCURACY[0] <= accuracy <= USABLE_ACCURACY[1]:
        notes.append(
            "WARNING: that leaves one class very thin. A correctness probe needs "
            "both right and wrong answers; see the note on MODEL in the "
            "configuration cell. Stopping now costs little."
        )
    return notes


def run_greedy_pass(
    model,
    params,
    sampler,
    tokenizer,
    questions: list[dict],
    *,
    shard_dir: str | Path,
    shard_size: int,
    gen_batch: int,
    extract_batch_size: int,
    max_new_tokens: int,
    terminators: set[int],
    main_dataset: str = "triviaqa",
    log=print,
) -> GreedyResult:
    """Answer every question greedily and record both residual positions.

    Resumes from whatever is already checkpointed in `shard_dir`.
    """
    shards = ShardWriter(shard_dir, shard_size=shard_size)
    start = shards.resume_at()
    total = len(questions)
    if start:
        log(f"Resuming the greedy pass after {start} of {total} questions.")
    if start > total:
        raise RuntimeError(
            f"{start} questions are checkpointed in {shard_dir} but only {total} were "
            "sampled; the question sample changed under an existing run"
        )

    progress = Progress(total, done_already=start)
    correct_so_far: list[bool] = []
    notes: list[str] = []
    max_abs = 0.0
    clipped = 0

    for block_start in range(start, total, gen_batch):
        block = questions[block_start : block_start + gen_batch]
        prompts = [build_prompt(q["question"]) for q in block]
        texts, answer_ids = generate(
            sampler, prompts, max_new_tokens=max_new_tokens, terminators=terminators
        )
        prompt_ids = [tokenizer.encode(p, add_bos=True) for p in prompts]

        # Extraction runs in smaller chunks than generation: its peak memory is
        # set by a vocabulary-sized logits tensor, generation's is not.
        for chunk in range(0, len(block), extract_batch_size):
            upto = chunk + extract_batch_size
            signals = extract_batch(
                model,
                params,
                prompt_ids[chunk:upto],
                answer_ids[chunk:upto],
                max_answer_tokens=max_new_tokens,
            )
            before16, clipped_before = to_float16(signals.before)
            after16, clipped_after = to_float16(signals.after)
            clipped += clipped_before + clipped_after
            max_abs = max(
                max_abs,
                float(np.abs(signals.before).max()),
                float(np.abs(signals.after).max()),
            )
            for j, question in enumerate(block[chunk:upto]):
                if question["dataset"] == main_dataset:
                    correct_so_far.append(
                        is_correct(texts[chunk + j], question["aliases"])
                    )
                shards.add(
                    {
                        "qid": question["qid"],
                        "answer": texts[chunk + j],
                        "n_answer_tokens": int(signals.n_answer_tokens[j]),
                        "first_token_prob": float(signals.first_token_prob[j]),
                        "first_token_entropy": float(signals.first_token_entropy[j]),
                        "answer_logprob_mean": float(signals.answer_logprob_mean[j]),
                        "answer_logprob_sum": float(signals.answer_logprob_sum[j]),
                    },
                    before16[j],
                    after16[j],
                )

        progress.update(block_start + len(block) - start)
        if progress.processed >= 100 and (block_start // gen_batch) % 10 == 0:
            log(progress.line(" (greedy pass)"))
        if not notes and len(correct_so_far) >= EARLY_CHECK_AFTER:
            notes = _early_accuracy_note(correct_so_far, progress)
            for note in notes:
                log(f"  {note}")

    shards.close()
    generations, before, after = shards.read_all()
    if before is None:
        raise RuntimeError(f"no activations were written to {shard_dir}")

    expected = [q["qid"] for q in questions]
    if [g["qid"] for g in generations] != expected:
        raise RuntimeError(
            "the checkpointed questions are not the sampled questions in order; "
            f"delete {shard_dir} and run again"
        )

    log(f"Greedy pass done in {format_duration(progress.elapsed())}.")
    log(
        f"Largest residual magnitude {max_abs:.0f} against a float16 ceiling of "
        f"65504; {clipped} values clipped."
    )
    return GreedyResult(
        generations=generations,
        before=before,
        after=after,
        max_abs_residual=max_abs,
        values_clipped=clipped,
        elapsed_seconds=progress.elapsed(),
        notes=notes,
    )


def run_selfconsistency_pass(
    sampler,
    questions: list[dict],
    greedy_answers: dict[str, str],
    *,
    shard_dir: str | Path,
    shard_size: int,
    gen_batch: int,
    max_new_tokens: int,
    terminators: set[int],
    n_samples: int,
    seed: int,
    log=print,
) -> list[dict]:
    """Sample `n_samples` answers per question and score agreement with greedy.

    Agreement is measured after the same normalization the correctness labels
    use, so "Lima" and "lima." count as the same answer.
    """
    shards = ShardWriter(shard_dir, shard_size=shard_size, with_arrays=False)
    start = shards.resume_at()
    total = len(questions)
    if start:
        log(f"Resuming after {start} of {total} self-consistency questions.")
    progress = Progress(total, done_already=start)

    for block_start in range(start, total, gen_batch):
        block = questions[block_start : block_start + gen_batch]
        prompts = [build_prompt(q["question"]) for q in block]
        samples: list[list[str]] = [[] for _ in block]
        for draw in range(n_samples):
            texts, _ = generate(
                sampler,
                prompts,
                max_new_tokens=max_new_tokens,
                terminators=terminators,
                rng=seed + draw,
            )
            for i, text in enumerate(texts):
                samples[i].append(text)

        for question, drawn in zip(block, samples, strict=True):
            reference = normalize_answer(greedy_answers[question["qid"]])
            agree = sum(normalize_answer(s) == reference for s in drawn)
            shards.add(
                {
                    "qid": question["qid"],
                    "agreement": agree / n_samples,
                    "n_samples": n_samples,
                    "samples": drawn,
                }
            )

        progress.update(block_start + len(block) - start)
        if progress.processed >= 100 and (block_start // gen_batch) % 10 == 0:
            log(progress.line(" (self-consistency)"))

    shards.close()
    records, _, _ = shards.read_all()
    log(f"Self-consistency done in {format_duration(progress.elapsed())}.")
    return records
