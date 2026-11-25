# results

Everything `aprobe` writes, and everything the README's generated block reads.
All of it is committed, so any number in the README can be traced to the file
it came from.

Empty until [`notebooks/extract.ipynb`](../notebooks/extract.ipynb) has been run
and its output unzipped to `data/raw/run/`.

| File | Written by | What it holds |
|---|---|---|
| `splits.json` | `label` | The train/val/test/shift assignment, with the seed and counts. |
| `labels.csv` | `label` | One row per question: answer, correctness, match rule, refusal, every confidence signal. |
| `accuracy.csv` | `label` | Accuracy and refusal rate per dataset and split. |
| `data_summary.csv` | `label` | Dataset names, versions, licenses and sample sizes, copied from the run. |
| `run_manifest.json` | `label` | Checkpoint, dtype, prompt, library versions, residual magnitudes. |
| `label_agreement.json` | `label` | Agreement with the hand grades, and every disagreement. |
| `layer_scan.csv` | `train` | Validation/test/shift AUROC for both probes at every layer and position. |
| `selection.json` | `train` | The layer and penalty chosen on validation, and the baseline weights. |
| `controls.csv` | `train` | The shuffled-label control, one row per permutation. |
| `scores.csv` | `train` | Per-question scores for every probe and baseline. `evaluate` reads only this. |
| `metrics.csv` | `evaluate` | AUROC, AUPRC and accuracy at 80%/50% coverage, with bootstrap intervals. |
| `comparisons.csv` | `evaluate` | Paired bootstrap of the probe against every other signal. |
| `risk_coverage.csv` | `evaluate` | The selective-answering curves behind the figure. |
| `calibration.csv`, `calibration_bins.csv` | `evaluate` | Expected calibration error and the reliability bins. |
| `excluding_refusals.csv` | `evaluate` | The same AUROCs on questions where the model committed to an answer. |
| `error_concentration.csv` | `evaluate` | Error rates by answer type, question length, refusal and match rule. |
| `examples_*.csv` | `evaluate` | The questions where the probe and the confidence signal most disagree. |
| `layer_curve.png`, `risk_coverage.png`, `calibration.png` | `evaluate` | The three figures. |
