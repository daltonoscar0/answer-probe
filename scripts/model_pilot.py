"""Measure a model's TriviaQA accuracy on a small sample, on CPU.

This is the evidence behind the choice of model. A probe that predicts "this
answer is wrong" needs both classes present; a model that is wrong on every
question gives it nothing to separate.

    python scripts/model_pilot.py gemma3-270m-it
    python scripts/model_pilot.py gemma3-1b-it --n 48

Needs the `extract` extra. It downloads the checkpoint (420 MB for the 270M,
~2 GB for the 1B) and runs on CPU; 48 questions takes well under a minute once
the weights are local. The sample is the first `n` questions of the same
seeded TriviaQA sample the real run uses, so it is a preview of that run and
not a separate draw.

Nothing it prints is a result: it is far too small, and it exists only to catch
a model that cannot support the study before a GPU session is spent on it.
"""

from __future__ import annotations

import argparse
import time

from answer_probe.extract import (
    CHECKPOINTS,
    build_sampler,
    generate,
    load_model,
    terminator_ids,
)
from answer_probe.normalize import is_refusal, match_answer
from answer_probe.prompts import build_prompt

USABLE = (0.10, 0.90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(CHECKPOINTS), nargs="?", default="gemma3-1b-it")
    parser.add_argument("--n", type=int, default=48, help="How many questions to try.")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--cache-dir", default="checkpoints")
    parser.add_argument("--quiet", action="store_true", help="Totals only.")
    args = parser.parse_args()

    import datasets

    from answer_probe.data import TRIVIAQA, _triviaqa_record, choose_indices

    split = datasets.load_dataset(TRIVIAQA.hf_path, TRIVIAQA.hf_config, split=TRIVIAQA.split)
    sample = choose_indices(len(split), TRIVIAQA.n_sample, TRIVIAQA.seed)
    # Spread the preview across the whole sample. Taking the first n would take
    # the n lowest row numbers, and TriviaQA's validation split is ordered by
    # question source, so that previews one corner of the data rather than it.
    stride = max(1, len(sample) // args.n)
    indices = sample[::stride][: args.n]
    questions = [
        _triviaqa_record(row, int(i))
        for row, i in zip(split.select(indices), indices, strict=True)
    ]

    model, params, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
    sampler = build_sampler(
        model, params, tokenizer,
        max_new_tokens=32, cache_length=192, pad_length=(128,),
    )
    terminators = terminator_ids(tokenizer)

    started = time.time()
    answers: list[str] = []
    for i in range(0, len(questions), args.batch):
        batch = questions[i : i + args.batch]
        texts, _ = generate(
            sampler,
            [build_prompt(q.question) for q in batch],
            max_new_tokens=32,
            terminators=terminators,
        )
        answers.extend(texts)
    elapsed = time.time() - started

    correct = refused = 0
    for question, answer in zip(questions, answers, strict=True):
        match = match_answer(answer, list(question.aliases))
        correct += match.correct
        refused += is_refusal(answer)
        if not args.quiet:
            mark = "OK " if match.correct else ("REF" if is_refusal(answer) else "  x")
            print(
                f"{mark} {question.question[:58]:60s} -> {answer[:30]!r:32s} "
                f"gold={list(question.aliases)[:2]}"
            )

    accuracy = correct / len(questions)
    print(
        f"\n{args.model}: {correct}/{len(questions)} = {accuracy:.1%} correct, "
        f"{refused} refusals, {elapsed:.0f}s on CPU"
    )
    if not USABLE[0] <= accuracy <= USABLE[1]:
        print(
            "This model cannot support the study: with one class this thin there is "
            "nothing for a correctness probe to separate, and nothing to evaluate it on."
        )
    else:
        print("Usable label balance for a correctness probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
