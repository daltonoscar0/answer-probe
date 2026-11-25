# answer-probe

Before a language model answers a factual question, do its hidden states already
say whether the answer will be wrong? This repository asks that question of
Gemma 3 on TriviaQA, with everything in JAX: inference through Google DeepMind's
[`gemma`](https://github.com/google-deepmind/gemma) library, probes trained with
Optax. For every question it records the residual stream at **every layer** at
two positions — the last token of the prompt, *before any answer token exists*,
and the last token of the answer — then asks whether a linear probe on those
states predicts errors better than the model's own confidence (first-token
probability, answer log-probability, entropy, and 8-sample self-consistency).
The probe is trained only on TriviaQA and tested again on Natural Questions to
see whether what it learned transfers.

**Either answer is a result.** A probe that beats confidence is useful for
reliability; one that does not is worth knowing. The analysis is deliberately
built so that it cannot be tuned toward the first outcome: the best layer and
the regularisation strength are chosen on a validation split, the test split is
touched once, and the README's numbers are written by `aprobe report --write`
rather than typed.

## Status

The code is complete and tested; **the extraction run has not been executed
yet**. [`notebooks/extract.ipynb`](notebooks/extract.ipynb) needs a Colab GPU
session, which only Oscar can start. Until it has run there are no results, and
none are invented here.

What *is* verified, offline and in CI:

- the extraction code, run on a randomly initialised Gemma 3 of toy size,
  including the check that the "before answering" state really is uncontaminated
  by the answer, and that batching and padding do not change a question's result;
- both notebook passes end to end — generation, extraction, checkpointing,
  resume, and the handover folder — feeding the analysis CLI;
- the full `label → train → evaluate → report` pipeline on a synthetic run with
  a planted signal;
- every metric against values worked out by hand, and the probes against a
  planted direction with the shuffled-label control beside them.

<!-- BEGIN generated: aprobe report -->

> **No extraction run yet.** Every number and figure in this section is written
> here by `aprobe report --write`, which reads `results/`. Until
> [`notebooks/extract.ipynb`](notebooks/extract.ipynb) has been run on Colab and
> its output unzipped to `data/raw/run/`, there is nothing to report, and
> nothing is filled in by hand.
>
> Once it has run, this block holds: the headline table (the before-answering
> probe against each confidence baseline, AUROC with a 95% bootstrap interval
> and accuracy at 80% coverage, on both the in-distribution test split and the
> shift set), the layer curve, the paired comparison against the strongest
> baseline, the shuffled-label and question-surface controls, calibration, and
> the ten questions each way where the probe and the confidence signal
> disagree most.

<!-- END generated: aprobe report -->

## Reproducing it

### 1. The GPU half, on Colab

Open [`notebooks/extract.ipynb`](notebooks/extract.ipynb) in Colab, choose
**Runtime → Change runtime type → T4 GPU**, and **Run all**. It mounts Google
Drive (one permission prompt), samples the questions, generates every answer
greedily, extracts the hidden states, and then runs the self-consistency pass.
It checkpoints to Drive every 250 questions, so a disconnected session is
resumed by opening it again and running all: it picks up from the last
checkpoint rather than starting over. It prints a rate and a time-remaining
estimate as it goes, and an accuracy read after the first 200 questions.

**About the Hugging Face token.** The brief assumed one would be needed. It is
not, for the weights: the JAX `gemma` library distributes Gemma from the public
`gs://gemma-data` bucket rather than the Hub, and the notebook fetches the
checkpoint from there over plain HTTPS with no credentials. If `HF_TOKEN` is in
Colab secrets the notebook reads it with `userdata.get("HF_TOKEN")` and uses it
for the dataset downloads only; it is never printed, never written to a file,
and a test asserts as much about the notebook's source. Both datasets are public,
so the run works without it.

The last cell writes `answer_probe_out.zip`. Unzip it so the folder lands at
`data/raw/run/` in a clone of this repository.

### 2. The analysis half, on a laptop

```bash
git clone https://github.com/daltonoscar0/answer-probe && cd answer-probe
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

aprobe label              # correctness labels, accuracy, and the blind grading sheet
aprobe grade              # grade 60 answers by hand, one at a time
aprobe label              # again: now it reports agreement with your grades
aprobe train              # probes at every layer and position, controls, baselines
aprobe evaluate           # metrics, intervals and the three figures -> results/
aprobe report --write     # the headline table, spliced into this README
```

CPU only, a few minutes end to end. Everything the commands write lands in
[`results/`](results/), and `aprobe evaluate` never refits anything — it reads
`results/scores.csv`, so the reported numbers can be recomputed, and checked,
from a file in the repository.

To see the pipeline run without a GPU or a notebook:

```bash
python scripts/smoke.py /tmp/aprobe-smoke   # synthetic run, meaningless numbers
```

## The model

`gemma3-1b-it`, bfloat16, from
`gs://gemma-data/checkpoints/gemma3-1b-it` — 26 blocks, hidden size 1152, so 28
recorded layers (layer 0 is the embedding output, 1–26 are block outputs, 27 is
the final RMSNorm). The exact checkpoint, library versions and prompt are
recorded in `results/run_manifest.json` by `aprobe label`.

**A deviation, stated plainly.** The brief asks for the smallest instruction-tuned
Gemma the library supports, which is `gemma3-270m-it`. We default to the 1B
instead. On closed-book trivia the 270M model is wrong on nearly every question,
and a probe that predicts "this answer is wrong" needs both classes present —
to train on, and to be evaluated against with usable intervals. The notebook
reports the accuracy it actually gets after 200 questions and warns if it falls
outside 10–90%, and `MODEL = "gemma3-270m-it"` at the top reproduces the
literal reading. A T4 is Turing and has no bfloat16 tensor cores, so XLA runs
these matmuls through float32: correct, but without the speedup newer hardware
would give.

Hidden states are stored as float16. The residual stream has large outliers —
measured at 2.2×10⁴ on the 270M model against a float16 ceiling of 6.55×10⁴ —
so values are clipped rather than allowed to overflow to infinity, and both the
largest magnitude seen and the number of clipped values go into the manifest.

## Data

| | Main set | Shift set |
|---|---|---|
| Dataset | TriviaQA, `rc.nocontext` | Natural Questions (open) |
| Split | `validation` (17,944 rows) | `validation` (3,610 rows) |
| Sampled | 6,000, seed 20260101 | 2,000, seed 20260102 |
| Used for | train 70% / val 15% / test 15% | held out entirely |
| License | Apache-2.0 per the TriviaQA repository (the HF card says "unknown") | CC BY-SA 3.0 |

NQ-open was used rather than PopQA: it is ungated, its licensing is explicit,
and it ships answer lists in the same shape. Exact row counts, dataset versions
and alias statistics for the actual run are written to
`results/data_summary.csv`.

Splits are keyed on question id with a fixed seed, saved to
`results/splits.json`, and checked to be a partition — no question appears in
two splits, and the shift set never enters training or model selection. The
self-consistency pass covers the validation, test and shift questions: the brief
asked for 1,500 TriviaQA test questions, but a 15% test split of 6,000 is only
900, so instead of sampling a subset we cover the test split completely and add
validation, which is where the combination model is fitted.

## Labels

An answer is correct if its SQuAD-normalized form equals a normalized gold
alias, or if a gold alias appears as a whole-word span inside a short (≤10 word)
answer. Two departures from the standard recipe, both because TriviaQA's alias
lists are inconsistent: apostrophes are deleted so "Dog's" matches "Dogs", and
other punctuation becomes a space so "Jean-Luc" matches "Jean Luc". Number words
are canonicalised to digits, so "nineteen sixty nine" and "1969" are the same
answer.

**The labels are audited.** `aprobe label` draws 60 answers stratified by
automatic label and writes them to a grading sheet that deliberately omits that
label; `aprobe grade` walks through them one at a time; the agreement is
reported and saved to `results/label_agreement.json`. If agreement is below 90%
the matcher is wrong and gets fixed before anything else is believed. The grades
live in `data/human_labels.jsonl`.

## Probes and controls

At every layer and both positions:

- **Linear probe** — logistic regression on standardized hidden states, L2
  penalty, trained with Optax. Deterministic: weights start at zero, the bias at
  the training base rate's logit, full batch, no shuffling. The penalty is
  chosen from six values on the validation split.
- **Difference of means** — project onto the difference between the mean states
  of wrong and right answers. No training beyond the two means, and deliberately
  not converted into a probability, since doing so would mean fitting something.

Reported alongside, all of them:

- **Shuffled labels** — the same probe refitted on permuted training labels and
  scored out of sample, averaged over five permutations. Out of sample because a
  1152-feature probe can partly memorise permuted labels in sample; averaged
  because a single permutation is noisy — a noise direction aligns with the real
  one by about 1/√D by chance.
- **Question-surface baseline** — logistic regression on the question text alone:
  length, whether it contains a digit, whether the gold answer is a number, and a
  one-hot answer type guessed from the question word.
- **Confidence baselines** — first-token probability, length-normalised answer
  log-probability, first-token entropy, and 8-sample self-consistency at
  temperature 1.0.
- **Refusals excluded** — the prompt allows "unknown" and a refusal is always
  scored wrong, so the headline signals are also scored on just the questions
  where the model committed to an answer.

Evaluation uses AUROC and AUPRC for predicting "the answer is wrong", with 95%
percentile bootstrap intervals resampled over questions, and a **paired**
bootstrap for probe-versus-baseline differences. Selective answering is reported
as a risk–coverage curve with accuracy at 80% and 50% coverage; tied scores are
credited with their tie group's error rate rather than the order they sit in the
file, which matters because self-consistency takes only nine distinct values.

## Limitations

- **One small model.** Everything here is one instruction-tuned Gemma 3 at one
  size. Nothing says the same directions exist, or sit at the same depth, in a
  larger model or a different family.
- **Two datasets, one shift.** TriviaQA and NQ-open are both short-answer,
  English, entity-centric QA. "Transfers to NQ-open" is a much weaker claim than
  "transfers".
- **Exact-match labels.** Correctness is alias matching, not judgement. It
  mislabels paraphrases, partially-right answers and answers that are right
  about a fact the gold list does not include. The human audit measures how
  often, but does not fix it.
- **Linear probes only.** A linear probe detects what is *linearly available* in
  the residual stream at one position. That is the whole claim. It says nothing
  about mechanism: finding a direction does not show the model uses it, does not
  show it causes the error, and does not show the information would survive any
  intervention. A probe can also read off a correlate of difficulty rather than
  anything about the model's state — which is what the question-surface baseline
  is there to bound. Read the headline as "this much is linearly decodable
  before the answer exists", not as "the model knows it is about to be wrong".
- **One prompt.** A single short-answer instruction, unchanged across questions
  and datasets. Error rates and probe accuracy both move with prompt wording.

## Repository

```
src/answer_probe/
  prompts.py     the one prompt, and the Gemma 3 chat template
  data.py        dataset specs and reproducible question sampling
  splits.py      question-level splits, checked to be a partition
  extract.py     residual-stream recording, generation, checkpoint download
  pipeline.py    the notebook's two GPU passes
  shards.py      resumable checkpointing for a session that may be cut off
  normalize.py   answer normalization and alias matching
  labeling.py    correctness labels and the human-agreement audit
  probes.py      the linear probe (Optax) and the difference-of-means probe
  features.py    question-surface features, answer types, confidence signals
  metrics.py     AUROC, AUPRC, bootstrap, risk-coverage, calibration
  train.py       fit everything, select on validation, save per-question scores
  evaluate.py    metrics and comparisons, computed from the saved scores
  analysis.py    where errors concentrate, and the disagreement examples
  plots.py       the three figures
  report.py      the headline table, and the README's generated block
  cli.py         aprobe label / grade / train / evaluate / report
notebooks/extract.ipynb    generated by scripts/build_notebook.py
scripts/smoke.py           the whole pipeline on a synthetic run
```

## Credits

- **TriviaQA** — Joshi, Choi, Weld & Zettlemoyer (2017), *TriviaQA: A Large Scale
  Distantly Supervised Challenge Dataset for Reading Comprehension*, ACL.
- **Natural Questions** — Kwiatkowski et al. (2019), *Natural Questions: A
  Benchmark for Question Answering Research*, TACL; the open-domain variant from
  Lee, Chang & Toutanova (2019), ACL.
- **Gemma** — Gemma Team, Google DeepMind. Weights are used under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms); the
  [`gemma`](https://github.com/google-deepmind/gemma) library is Apache-2.0.
- Probing hidden states for truthfulness and error is an established line of
  work; this is a small, carefully controlled replication-style study in that
  vein rather than a new method.

## License

MIT — see [LICENSE](LICENSE). The license covers this code, not the datasets or
the model weights, which keep their own terms.
