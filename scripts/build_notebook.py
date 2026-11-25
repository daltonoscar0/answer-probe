"""Generate notebooks/extract.ipynb.

The notebook is written here rather than edited in place so it stays reviewable
as plain Python and cannot pick up stray execution counts or outputs. CI checks
that regenerating it produces no diff.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "extract.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(
    """
# answer-probe: generate answers and extract hidden states

Runs Gemma 3 on TriviaQA and NQ-open, saves what it answered, and saves the
residual stream at every layer at two positions: the **last token of the
prompt**, before any answer token exists, and the **last token of the answer**.
Everything downstream (labels, probes, metrics, plots) runs on a laptop from the
folder this notebook produces.

## How to run it

1. **Runtime → Change runtime type → T4 GPU.**
2. **Runtime → Run all.** Google Drive will ask for permission once, near the
   top; that is the only click.
3. Wait. The run checkpoints to Drive every few hundred questions, so if Colab
   disconnects, just open the notebook again and **Run all** — it resumes from
   the last checkpoint rather than starting over.
4. The last cell writes `answer_probe_out.zip`. Download it and unzip it into
   the repository at `data/raw/run/`.

If the install cell says a restart is needed, use **Runtime → Restart session
and run all**. Nothing is lost: the checkpoints are on Drive.

## What it does not need

No Hugging Face token is required for the weights. The JAX `gemma` library
distributes Gemma from a public Google Cloud Storage bucket, not from the Hub,
and this notebook fetches it over plain HTTPS. A token is read from Colab
secrets if one is there, and used only for the dataset downloads.
"""
)

code(
    '''
# --- Configuration. These are the only knobs; nothing below needs editing. ---

# Which model to run. The project brief asks for the smallest instruction-tuned
# Gemma the JAX library supports, which is gemma3-270m-it. We default to
# gemma3-1b-it instead: on closed-book trivia the 270M model is wrong almost
# every time, and a correctness probe needs enough correct answers to learn
# from and to be evaluated against. Set MODEL = "gemma3-270m-it" to run the
# smaller one; the notebook reports the accuracy it is actually getting after
# the first 200 questions, and again at the end of the greedy pass.
MODEL = "gemma3-1b-it"

# bfloat16 is what Gemma was trained in, and what the activation files store,
# so the saved residual stream is exactly the one the model computed. A T4 is
# Turing and has no bfloat16 tensor cores, so XLA runs these matmuls through
# float32 -- correct, but without the speedup an A100 would give.
MODEL_DTYPE = "bfloat16"

MAX_NEW_TOKENS = 32
GEN_BATCH = 32        # prompts answered per sampler call
EXTRACT_BATCH = 8     # questions per forward pass; bounded by the vocab-sized
                      # logits tensor, so lower this first if you hit OOM
SHARD_SIZE = 250      # questions per checkpoint written to Drive

N_SELFCONS_SAMPLES = 8
SELFCONS_TEMPERATURE = 1.0
SELFCONS_SEED = 20260107

DRIVE_ROOT = "/content/drive/MyDrive/answer-probe"
OUTPUT_NAME = "answer_probe_out"
'''
)

md("## 1. Install")

code(
    '''
# gemma pulls in kauldron, orbax and tensorflow-cpu. protobuf is pinned up
# because sentencepiece ships generated code that needs a 7.x runtime.
%pip install -q "gemma" "protobuf>=7.36.1" "datasets>=2.19"

# Colab ships a CUDA jax; installing gemma can replace it with a CPU-only
# build, so put the CUDA one back. This is a no-op when it is already right.
%pip install -q -U "jax[cuda12]"

# The shared code: the prompt, the sampling seeds, the extraction function and
# the split logic all live in the repository, so the notebook and the analysis
# run the same definitions.
%pip install -q "answer-probe @ git+https://github.com/daltonoscar0/answer-probe.git"
'''
)

code(
    '''
import json, os, time, shutil
from pathlib import Path

import numpy as np

# Must happen before the gemma import on any platform without 128-bit floats.
from answer_probe.extract import patch_numpy_long_double
patch_numpy_long_double()

import jax
import answer_probe
from gemma import gm

print("answer-probe", answer_probe.__version__)
print("jax", jax.__version__, "| devices:", jax.devices())

if not any(d.platform == "gpu" for d in jax.devices()):
    raise SystemExit(
        "No GPU visible to JAX.\\n"
        "Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all.\\n"
        "If a GPU is selected and this still fails, the jax[cuda12] install "
        "above needs a restart: Runtime -> Restart session and run all."
    )
'''
)

md("## 2. Drive, for checkpoints that survive a disconnect")

code(
    '''
from google.colab import drive

drive.mount("/content/drive")

RUN_ROOT = Path(DRIVE_ROOT) / MODEL
CKPT_GREEDY = RUN_ROOT / "greedy"
CKPT_SELFCONS = RUN_ROOT / "selfconsistency"
WEIGHTS_CACHE = Path(DRIVE_ROOT) / "weights"
for directory in (CKPT_GREEDY, CKPT_SELFCONS, WEIGHTS_CACHE):
    directory.mkdir(parents=True, exist_ok=True)
print("checkpoints:", RUN_ROOT)
'''
)

code(
    '''
# A Hugging Face token, if one is in Colab secrets, for the dataset downloads.
# Never printed, never written to disk by this notebook. The model weights do
# not need it.
try:
    from google.colab import userdata

    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN found in Colab secrets and set for this session.")
    del token
except Exception as exc:
    print(f"No HF_TOKEN available ({type(exc).__name__}); both datasets are public, continuing.")
'''
)

md("## 3. Sample the questions")

code(
    '''
import pandas as pd

from answer_probe import data as apdata
from answer_probe.splits import make_splits, save_splits, selfconsistency_qids

QUESTIONS_PATH = RUN_ROOT / "questions.jsonl"
SUMMARY_PATH = RUN_ROOT / "data_summary.csv"

if QUESTIONS_PATH.exists():
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line]
    summary = pd.read_csv(SUMMARY_PATH)
    print(f"Reusing the sampled questions from Drive ({len(questions)}).")
else:
    sampled, summaries = apdata.sample_all()
    questions = [apdata.question_to_dict(q) for q in sampled]
    summary = pd.DataFrame(summaries)
    QUESTIONS_PATH.write_text(
        "\\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\\n"
    )
    summary.to_csv(SUMMARY_PATH, index=False)

qids = [q["qid"] for q in questions]
datasets = [q["dataset"] for q in questions]
splits = make_splits(qids, datasets)
save_splits(splits, RUN_ROOT / "splits.json")
selfcons_qids = [q for q in selfconsistency_qids(splits)]

print(f"{len(questions)} questions: " + ", ".join(
    f"{name} {len(group)}" for name, group in splits.items()
))
print(f"{len(selfcons_qids)} of them get the self-consistency pass.")
display(summary)
'''
)

md("## 4. Load the model")

code(
    '''
from answer_probe.extract import (
    BF16_MAX,
    build_sampler,
    generate,
    load_model,
    n_layers_recorded,
    terminator_ids,
)
from answer_probe.prompts import INSTRUCTION, PROMPT_VERSION, build_prompt
from answer_probe.extract import CHECKPOINTS
from answer_probe.shards import format_duration

t0 = time.time()
model, params, tokenizer = load_model(MODEL, dtype=MODEL_DTYPE, cache_dir=WEIGHTS_CACHE)
print(f"Loaded {MODEL} in {time.time() - t0:.0f}s")

N_LAYERS = n_layers_recorded(model)
HIDDEN = model.config.embed_dim
TERMINATORS = terminator_ids(tokenizer)
print(f"{model.config.num_layers} blocks -> {N_LAYERS} recorded layers, hidden size {HIDDEN}")
print("checkpoint:", CHECKPOINTS[MODEL]["path"])

# One cache for the greedy pass; the prompts are short, so the library's 4096
# default would allocate a KV cache far bigger than anything here needs.
CACHE_LENGTH = 256
PAD_LENGTH = (128, 256)
greedy_sampler = build_sampler(
    model, params, tokenizer,
    max_new_tokens=MAX_NEW_TOKENS, cache_length=CACHE_LENGTH, pad_length=PAD_LENGTH,
)
'''
)

code(
    '''
# Sanity check on one question: the prompt the model actually sees, where it
# ends, and what comes back.
example = build_prompt(questions[0]["question"])
example_ids = tokenizer.encode(example, add_bos=True)
print(repr(example))
print(f"{len(example_ids)} tokens; last three decode to "
      f"{tokenizer.decode(example_ids[-3:])!r}")

texts, ids = generate(
    greedy_sampler, [example], max_new_tokens=MAX_NEW_TOKENS, terminators=TERMINATORS
)
print("\\nQ:", questions[0]["question"])
print("A:", repr(texts[0]))
print("gold:", questions[0]["aliases"][:4])
'''
)

md(
    """
## 5. Greedy pass: answers, hidden states and confidence signals

One sampler call per batch of prompts, then one teacher-forced forward pass over
prompt + answer. Attention is causal, so the state at the last prompt token in
that pass is exactly the state the model had before it produced anything — the
"before answering" position costs no extra compute.

This pass finishes and is saved completely before the expensive self-consistency
pass starts, so a session that runs out of time still leaves usable data. The
loop itself is `answer_probe.pipeline.run_greedy_pass`, which has tests that run
it end to end on a tiny model; what is left here is the configuration.
"""
)

code(
    '''
from answer_probe.pipeline import run_greedy_pass, run_selfconsistency_pass

greedy = run_greedy_pass(
    model, params, greedy_sampler, tokenizer, questions,
    shard_dir=CKPT_GREEDY,
    shard_size=SHARD_SIZE,
    gen_batch=GEN_BATCH,
    extract_batch_size=EXTRACT_BATCH,
    max_new_tokens=MAX_NEW_TOKENS,
    terminators=TERMINATORS,
)

generations = greedy.generations
hidden_before, hidden_after = greedy.before, greedy.after
print(f"\\n{len(generations)} answers, activations {hidden_before.shape} {hidden_before.dtype}")
'''
)

md("## 6. Check the accuracy and the label balance")

code(
    '''
from answer_probe.normalize import is_correct, is_refusal

by_qid = {q["qid"]: q for q in questions}
correct = np.array([
    is_correct(g["answer"], by_qid[g["qid"]]["aliases"]) for g in generations
])
refused = np.array([is_refusal(g["answer"]) for g in generations])
is_main = np.array([by_qid[g["qid"]]["dataset"] == "triviaqa" for g in generations])

for name, mask in (("TriviaQA", is_main), ("NQ-open", ~is_main)):
    print(f"{name}: {correct[mask].mean():.1%} correct, {refused[mask].mean():.1%} refused "
          f"({mask.sum()} questions)")

for row in generations[:8]:
    gold = by_qid[row["qid"]]["aliases"][0]
    mark = "OK " if is_correct(row["answer"], by_qid[row["qid"]]["aliases"]) else "   "
    print(f"  {mark} {by_qid[row['qid']]['question'][:62]:64s} -> "
          f"{row['answer'][:28]!r:30s} (gold {gold[:24]!r})")

accuracy = correct[is_main].mean()
if not 0.10 <= accuracy <= 0.90:
    print(
        f"\\nWARNING: TriviaQA accuracy is {accuracy:.1%}. A correctness probe needs both "
        "classes represented; outside roughly 10-90% correct the estimates get very "
        "noisy.\\nConsider setting MODEL at the top to the next size up (gemma3-4b-it) "
        "or down (gemma3-270m-it) and running again. Checkpoints are kept per model, "
        "so nothing already computed is lost."
    )
else:
    print(f"\\nLabel balance is workable: {accuracy:.1%} correct on TriviaQA.")
'''
)

md(
    """
## 7. Self-consistency

Eight samples at temperature 1.0 per question, scored by how often they agree
with the greedy answer after normalization. Run only on the validation, test and
shift questions: it costs eight generations per question, and the probe is never
trained on those splits anyway.
"""
)

code(
    '''
sample_sampler = build_sampler(
    model, params, tokenizer,
    max_new_tokens=MAX_NEW_TOKENS, cache_length=CACHE_LENGTH, pad_length=PAD_LENGTH,
    temperature=SELFCONS_TEMPERATURE,
)

_selfcons_t0 = time.time()
selfcons = run_selfconsistency_pass(
    sample_sampler,
    [by_qid[q] for q in selfcons_qids],
    {g["qid"]: g["answer"] for g in generations},
    shard_dir=CKPT_SELFCONS,
    shard_size=SHARD_SIZE,
    gen_batch=GEN_BATCH,
    max_new_tokens=MAX_NEW_TOKENS,
    terminators=TERMINATORS,
    n_samples=N_SELFCONS_SAMPLES,
    seed=SELFCONS_SEED,
)
# Wall clock for this cell only. On a resumed run it covers what this session
# did, not the whole pass, and the summary below says so.
selfcons_elapsed = time.time() - _selfcons_t0
print(f"{len(selfcons)} self-consistency records of {len(selfcons_qids)} expected")
'''
)

md("## 8. Assemble the output folder")

code(
    '''
from answer_probe.io import write_run_folder

out_dir = Path("/content") / OUTPUT_NAME
if out_dir.exists():
    shutil.rmtree(out_dir)

manifest = {
    "model": MODEL,
    "model_class": CHECKPOINTS[MODEL]["class"],
    "checkpoint": CHECKPOINTS[MODEL]["path"],
    "model_dtype": MODEL_DTYPE,
    "prompt_version": PROMPT_VERSION,
    "instruction": INSTRUCTION,
    "max_new_tokens": MAX_NEW_TOKENS,
    "selfconsistency_samples": N_SELFCONS_SAMPLES,
    "selfconsistency_temperature": SELFCONS_TEMPERATURE,
    "selfconsistency_seed": SELFCONS_SEED,
    "selfconsistency_splits_covered": sorted({
        name for name, group in splits.items()
        if group and set(group) <= {r["qid"] for r in selfcons}
    }),
    "layer_indexing": "0 = embeddings, 1..N = block outputs, N+1 = final RMSNorm",
    "activation_dtype": "bfloat16",
    "max_abs_residual": greedy.max_abs_residual,
    "bfloat16_ceiling": BF16_MAX,
    "values_clipped": int(greedy.values_out_of_range),
    "answer_probe_version": answer_probe.__version__,
    "jax_version": jax.__version__,
    "gemma_version": getattr(gm, "__version__", "unknown"),
    "generated_on_gpu": str(jax.devices()[0]),
}

write_run_folder(
    out_dir,
    manifest=manifest,
    questions=questions,
    generations=generations,
    selfconsistency=selfcons,
    before=hidden_before,
    after=hidden_after,
    qids=[q["qid"] for q in questions],
    data_summary=summary,
    split_map=splits,
)
print(json.dumps(manifest, indent=2)[:1200])
'''
)

code(
    '''
archive = shutil.make_archive(f"/content/{OUTPUT_NAME}", "zip", out_dir.parent, out_dir.name)
size_mb = Path(archive).stat().st_size / 1e6
print(f"{archive}  ({size_mb:.0f} MB)")

# A copy on Drive too, so a dropped browser tab does not cost the whole run.
shutil.copy(archive, Path(DRIVE_ROOT) / f"{OUTPUT_NAME}.zip")

from google.colab import files
files.download(archive)
'''
)

md(
    """
## Run summary

Short enough to paste back into a chat or an issue, and it carries the things
worth knowing without the questions, the answers or anything else that only
matters to the pipeline. No token and no file path outside `/content` appears
in it.
"""
)

code(
    '''
# Everything here is already computed; this cell only formats it.
sc_per_q = N_SELFCONS_SAMPLES
lines = [
    "answer-probe run summary",
    f"  model            {MODEL} ({MODEL_DTYPE}), on {jax.devices()[0].device_kind}",
    f"  questions        {len(questions)} "
    f"({', '.join(f'{k} {len(v)}' for k, v in splits.items())})",
    f"  greedy answers   {len(generations)} at <={MAX_NEW_TOKENS} new tokens",
    f"  self-consistency {len(selfcons)} records "
    f"({len(selfcons) // sc_per_q if sc_per_q else 0} questions x {sc_per_q})",
    f"  accuracy         TriviaQA {correct[is_main].mean():.1%} correct, "
    f"{refused[is_main].mean():.1%} refused",
    f"                   NQ-open  {correct[~is_main].mean():.1%} correct, "
    f"{refused[~is_main].mean():.1%} refused",
    f"  activations      before {hidden_before.shape} {hidden_before.dtype}, "
    f"after {hidden_after.shape}; {N_LAYERS} recorded layers, hidden {HIDDEN}",
    f"  bfloat16 range   max |value| {greedy.max_abs_residual:.3g} "
    f"(ceiling {BF16_MAX:.3g}), {greedy.values_out_of_range} clipped",
    f"  this session     greedy {format_duration(greedy.elapsed_seconds)}, "
    f"self-consistency {format_duration(selfcons_elapsed)} "
    f"(a resumed run only times what it did itself)",
    f"  archive          {Path(archive).name}, {size_mb:.0f} MB",
    f"  versions         answer-probe {answer_probe.__version__}, "
    f"jax {jax.__version__}, gemma {getattr(gm, '__version__', 'unknown')}",
]
if greedy.values_out_of_range:
    lines.append("  WARNING: values were clipped; the activations are not faithful.")
print("\\n".join(lines))

(Path(DRIVE_ROOT) / "run_summary.txt").write_text("\\n".join(lines) + "\\n")
'''
)

md(
    """
## Next

Unzip the download into the repository so the folder lands at
`data/raw/run/`, then:

```bash
aprobe label      # correctness labels + the blind grading sheet
aprobe grade      # grade 60 answers by hand
aprobe label      # again, to report agreement
aprobe train      # probes at every layer, controls, baselines
aprobe evaluate   # metrics and figures into results/
aprobe report     # the headline table
```
"""
)


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": lines,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells)")
