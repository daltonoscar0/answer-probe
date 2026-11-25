# data

Nothing large is committed here. `.gitignore` excludes `data/raw/` and
`data/activations/`; everything in them is re-derivable by running
[`notebooks/extract.ipynb`](../notebooks/extract.ipynb) again.

```
data/
  raw/run/                      <- unzip answer_probe_out.zip here  (ignored)
    manifest.json
    questions.jsonl             the sampled questions and their gold aliases
    generations.jsonl           the greedy answer and confidence signals
    selfconsistency.jsonl       8 samples per evaluated question
    splits.json
    data_summary.csv
    activations/
      hidden_before.npy         bfloat16 [questions, layers, hidden], stored as uint16
      hidden_after.npy          bfloat16 [questions, layers, hidden], stored as uint16
      index.json                question id order of the array rows
  human_grading_sheet.jsonl     written by `aprobe label`, without the labels
  human_labels.jsonl            your grades, written by `aprobe grade`  (committed)
```

The two grading files are small and are the evidence behind the label-agreement
number, so `human_labels.jsonl` is committed once it exists.
