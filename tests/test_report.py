"""The README's numbers must come from `results/`, never from a keyboard.

`test_the_readme_block_matches_the_results` is the enforcement: as soon as a
real run exists, the committed README has to equal what `aprobe report`
renders from it. Before then, the block has to say there is nothing to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from answer_probe import io, report

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def committed_block() -> str:
    text = README.read_text()
    assert report.BEGIN_MARKER in text and report.END_MARKER in text
    return text.split(report.BEGIN_MARKER, 1)[1].split(report.END_MARKER, 1)[0]


def test_the_readme_has_a_generated_block():
    assert report.BEGIN_MARKER in README.read_text()
    assert report.END_MARKER in README.read_text()


def test_the_readme_block_matches_the_results():
    """Either a real run backs the block, or the block says there is no run."""
    results = io.results_dir()
    metrics_path = results / "metrics.csv"
    if not metrics_path.exists():
        block = committed_block()
        assert "No extraction run yet" in block, (
            "results/metrics.csv is missing, so the README must say so rather "
            "than show numbers"
        )
        assert "AUROC |" not in block, "the README shows a results table with no run behind it"
        return

    selection = json.loads((results / "selection.json").read_text())
    examples = {}
    for key in ("probe_caught", "confidence_caught"):
        path = results / f"examples_{key}_test.csv"
        if path.exists():
            examples[key] = pd.read_csv(path)
    agreement_path = results / "label_agreement.json"
    rendered = report.render_results_section(
        metrics=pd.read_csv(metrics_path),
        comparisons=pd.read_csv(results / "comparisons.csv"),
        controls=pd.read_csv(results / "controls.csv"),
        accuracy=pd.read_csv(results / "accuracy.csv"),
        selection=selection,
        calibration=pd.read_csv(results / "calibration.csv"),
        excluding_refusals=pd.read_csv(results / "excluding_refusals.csv"),
        agreement=json.loads(agreement_path.read_text()) if agreement_path.exists() else None,
        examples=examples,
    )
    assert committed_block().strip() == rendered.strip(), (
        "README is out of date; run `aprobe report --write`"
    )


def test_the_readme_states_the_model_deviation():
    """The departure from "smallest variant" has to be in the README, not just the code."""
    text = README.read_text()
    assert "gemma3-270m-it" in text
    assert "deviation" in text.lower()


def test_the_readme_is_honest_about_the_token():
    text = README.read_text()
    assert "userdata.get" in text
    assert "never printed" in text


def test_splice_replaces_only_the_generated_block(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        f"# Title\n\nintro\n\n{report.BEGIN_MARKER}\nold\n{report.END_MARKER}\n\ntail\n"
    )
    assert report.splice_into_readme(path, "new numbers\n")
    text = path.read_text()
    assert "# Title" in text and "intro" in text and "tail" in text
    assert "old" not in text and "new numbers" in text


def test_splice_is_idempotent(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(f"a\n{report.BEGIN_MARKER}\nx\n{report.END_MARKER}\nb\n")
    report.splice_into_readme(path, "same\n")
    assert report.splice_into_readme(path, "same\n") is False


def test_splice_refuses_a_readme_without_markers(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("no markers here\n")
    with pytest.raises(ValueError, match="generated block"):
        report.splice_into_readme(path, "numbers\n")


def test_headline_table_marks_the_headline_row():
    metrics = pd.DataFrame(
        [
            {
                "split": "test", "signal": "probe_before_linear",
                "label": "Linear probe, before answering", "error_rate": 0.4,
                "auroc": 0.7, "auroc_low": 0.6, "auroc_high": 0.8, "acc_at_80": 0.65,
            },
            {
                "split": "test", "signal": "conf_first_token_prob",
                "label": "First-token probability", "error_rate": 0.4,
                "auroc": 0.6, "auroc_low": 0.5, "auroc_high": 0.7, "acc_at_80": 0.62,
            },
        ]
    )
    table = report.headline_table(metrics, headline_signal="probe_before_linear")
    assert "**Linear probe, before answering**" in table
    assert "**0.700 [0.600, 0.800]**" in table
    assert "First-token probability | 0.600 [0.500, 0.700]" in table
    # The floor row: answering everything, no abstention.
    assert "No abstention" in table and "0.600" in table


def test_comparison_sentence_reads_the_interval_not_the_point():
    comparisons = pd.DataFrame(
        [
            {
                "split": "test", "reference": "probe_before_linear",
                "other": "conf_first_token_prob", "other_label": "First-token probability",
                "auroc_difference": 0.02, "auroc_difference_low": -0.01,
                "auroc_difference_high": 0.05,
                "reference_better": False, "indistinguishable": True,
            }
        ]
    )
    sentence = report.comparison_sentence(
        comparisons, split="test", other="conf_first_token_prob"
    )
    assert "indistinguishable" in sentence
    assert "+0.020" in sentence and "-0.010" in sentence


def test_comparison_sentence_when_the_probe_wins_and_loses():
    def frame(low, high, better, same):
        return pd.DataFrame(
            [{
                "split": "test", "reference": "p", "other": "c", "other_label": "c",
                "auroc_difference": (low + high) / 2,
                "auroc_difference_low": low, "auroc_difference_high": high,
                "reference_better": better, "indistinguishable": same,
            }]
        )

    assert "beats it" in report.comparison_sentence(
        frame(0.02, 0.08, True, False), split="test", other="c"
    )
    assert "loses to it" in report.comparison_sentence(
        frame(-0.09, -0.02, False, False), split="test", other="c"
    )


def test_controls_sentence_reports_the_range_not_just_the_mean():
    controls = pd.DataFrame(
        {
            "seed": [1, 2, 3],
            "linear_auroc_test": [0.48, 0.51, 0.52],
            "diffmeans_auroc_test": [0.49, 0.50, 0.51],
            "linear_auroc_train_insample": [0.62, 0.63, 0.61],
        }
    )
    sentence = report.controls_sentence(controls)
    assert "3 permutations" in sentence
    assert "0.503" in sentence and "0.480" in sentence and "0.520" in sentence
    assert "in sample" in sentence
