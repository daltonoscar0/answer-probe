"""End-to-end: label, train, evaluate and report on a synthetic run.

This is the test that catches schema drift between the notebook and the analysis
side, and it is the only place the four CLI commands run in sequence. The
numbers it produces are meaningless -- the fixture's activations are random with
a planted direction -- but the shapes, the file list, and the controls are real.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

import fake_run
from answer_probe import cli, io
from answer_probe.splits import SHIFT, TEST, TRAIN, VAL


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole pipeline once in a temporary directory."""
    root = tmp_path_factory.mktemp("pipeline")
    run_dir = root / "run"
    fake_run.write(run_dir, n_main=400, n_shift=150, seed=0)
    io.set_dirs(data=root / "data", results=root / "results")

    runner = CliRunner()
    outputs = {}
    for args in (
        ["label", "--run-dir", str(run_dir)],
        ["train", "--run-dir", str(run_dir)],
        ["evaluate", "--run-dir", str(run_dir), "--n-boot", "80"],
        ["report"],
    ):
        result = runner.invoke(cli.app, args)
        if result.exit_code != 0:
            raise AssertionError(
                f"`aprobe {' '.join(args)}` failed:\n{result.output}\n{result.exception!r}"
            )
        outputs[args[0]] = result.output
    yield root, outputs
    io.set_dirs(data=io.REPO_ROOT / "data", results=io.REPO_ROOT / "results")


def _results(root):
    return root / "results"


def test_every_expected_artefact_is_written(pipeline):
    root, _ = pipeline
    expected = [
        "splits.json",
        "labels.csv",
        "accuracy.csv",
        "data_summary.csv",
        "run_manifest.json",
        "label_agreement.json",
        "layer_scan.csv",
        "controls.csv",
        "scores.csv",
        "selection.json",
        "metrics.csv",
        "comparisons.csv",
        "risk_coverage.csv",
        "calibration.csv",
        "calibration_bins.csv",
        "excluding_refusals.csv",
        "error_concentration.csv",
        "layer_curve.png",
        "risk_coverage.png",
        "calibration.png",
    ]
    missing = [name for name in expected if not (_results(root) / name).exists()]
    assert not missing, f"missing results files: {missing}"


def test_splits_partition_the_questions_and_hold_back_the_shift_set(pipeline):
    root, _ = pipeline
    payload = json.loads((_results(root) / "splits.json").read_text())
    split_map = payload["splits"]

    all_qids = [q for qids in split_map.values() for q in qids]
    assert len(all_qids) == len(set(all_qids)) == 550

    assert all(q.startswith("nq_open:") for q in split_map[SHIFT])
    for name in (TRAIN, VAL, TEST):
        assert all(q.startswith("triviaqa:") for q in split_map[name])
    assert len(split_map[TRAIN]) == 280  # 70% of 400
    assert len(split_map[VAL]) == 60
    assert len(split_map[TEST]) == 60
    assert len(split_map[SHIFT]) == 150


def test_the_shift_set_never_enters_training(pipeline):
    root, _ = pipeline
    split_map = json.loads((_results(root) / "splits.json").read_text())["splits"]
    assert not set(split_map[SHIFT]) & set(split_map[TRAIN])
    scores = pd.read_csv(_results(root) / "scores.csv")
    # Only the evaluated splits appear in the scores file; train never does.
    assert set(scores["split"]) == {VAL, TEST, SHIFT}


def test_labels_cover_every_question_and_record_the_signals(pipeline):
    root, _ = pipeline
    labels = pd.read_csv(_results(root) / "labels.csv")
    assert len(labels) == 550
    assert set(labels["correct"].unique()) <= {0, 1}
    assert (labels["correct"] + labels["wrong"] == 1).all()
    for column in (
        "first_token_prob",
        "first_token_entropy",
        "answer_logprob_mean",
        "answer_type",
        "refused",
    ):
        assert column in labels.columns
    # "unknown" is one of the fixture's wrong answers, so refusals exist and are
    # all scored wrong.
    refusals = labels[labels["refused"] == 1]
    assert len(refusals) > 0
    assert (refusals["correct"] == 0).all()


def test_the_layer_scan_covers_every_layer_and_both_positions(pipeline):
    root, _ = pipeline
    scan = pd.read_csv(_results(root) / "layer_scan.csv")
    assert set(scan["position"]) == {"before", "after"}
    assert set(scan["probe"]) == {"linear", "diffmeans"}
    assert sorted(scan["layer"].unique()) == list(range(fake_run.N_LAYERS))
    assert len(scan) == fake_run.N_LAYERS * 2 * 2


def test_the_best_layer_is_the_best_on_validation_not_on_test(pipeline):
    root, _ = pipeline
    scan = pd.read_csv(_results(root) / "layer_scan.csv")
    selection = json.loads((_results(root) / "selection.json").read_text())["selection"]
    for key, choice in selection["per_probe"].items():
        del key
        rows = scan[(scan["position"] == choice["position"]) & (scan["probe"] == choice["probe"])]
        assert choice["layer"] == int(rows.loc[rows["auroc_val"].idxmax(), "layer"])
        assert choice["selected_on"] == "val"


def test_the_probe_finds_the_planted_signal(pipeline):
    """A sanity check on the pipeline, not a result: the fixture planted a signal."""
    root, _ = pipeline
    metrics = pd.read_csv(_results(root) / "metrics.csv")
    probe = metrics[(metrics["split"] == TEST) & (metrics["signal"] == "probe_before_linear")]
    assert len(probe) == 1
    assert probe.iloc[0]["auroc"] > 0.6
    assert probe.iloc[0]["auroc_low"] < probe.iloc[0]["auroc"] < probe.iloc[0]["auroc_high"]


def test_the_shuffled_label_control_lands_near_chance(pipeline):
    root, _ = pipeline
    controls = pd.read_csv(_results(root) / "controls.csv")
    assert len(controls) == 5
    assert controls["linear_auroc_test"].mean() == pytest.approx(0.5, abs=0.08)


def test_deeper_layers_carry_more_of_the_planted_signal(pipeline):
    """The fixture's signal grows with depth, so the layer curve should too."""
    root, _ = pipeline
    scan = pd.read_csv(_results(root) / "layer_scan.csv")
    rows = scan[(scan["position"] == "before") & (scan["probe"] == "linear")]
    rows = rows.sort_values("layer")
    assert rows.iloc[-1]["auroc_test"] > rows.iloc[0]["auroc_test"]


def test_metrics_cover_every_signal_on_both_evaluation_splits(pipeline):
    root, _ = pipeline
    metrics = pd.read_csv(_results(root) / "metrics.csv")
    assert set(metrics["split"]) == {TEST, SHIFT}
    for signal in (
        "probe_before_linear",
        "probe_after_linear",
        "probe_before_diffmeans",
        "conf_first_token_prob",
        "conf_answer_logprob",
        "conf_first_token_entropy",
        "conf_selfconsistency",
        "surface",
        "combination",
    ):
        assert (metrics["signal"] == signal).sum() == 2, signal
    assert ((metrics["auroc"] >= 0) & (metrics["auroc"] <= 1)).all()
    assert (metrics["auprc"] >= metrics["error_rate"] - 0.25).all()


def test_accuracy_at_coverage_beats_answering_everything(pipeline):
    root, _ = pipeline
    metrics = pd.read_csv(_results(root) / "metrics.csv")
    probe = metrics[(metrics["split"] == TEST) & (metrics["signal"] == "probe_before_linear")]
    row = probe.iloc[0]
    assert row["acc_at_80"] > 1 - row["error_rate"]
    assert row["acc_at_50"] >= row["acc_at_80"]


def test_comparisons_are_paired_and_cover_every_other_signal(pipeline):
    root, _ = pipeline
    comparisons = pd.read_csv(_results(root) / "comparisons.csv")
    assert (comparisons["reference"] == "probe_before_linear").all()
    assert "probe_before_linear" not in set(comparisons["other"])
    assert (
        comparisons["auroc_difference_low"] <= comparisons["auroc_difference"]
    ).all()
    assert (
        comparisons["auroc_difference"] <= comparisons["auroc_difference_high"]
    ).all()


def test_calibration_bins_account_for_every_question(pipeline):
    root, _ = pipeline
    bins = pd.read_csv(_results(root) / "calibration_bins.csv")
    scores = pd.read_csv(_results(root) / "scores.csv")
    for split in (TEST, SHIFT):
        expected = (scores["split"] == split).sum()
        assert bins[bins["split"] == split]["count"].sum() == expected
    calib = pd.read_csv(_results(root) / "calibration.csv")
    assert ((calib["ece"] >= 0) & (calib["ece"] <= 1)).all()


def test_risk_coverage_curve_is_monotone_in_coverage(pipeline):
    root, _ = pipeline
    rc = pd.read_csv(_results(root) / "risk_coverage.csv")
    assert not rc.empty
    assert rc["coverage"].between(0, 1).all()
    assert rc["accuracy"].between(0, 1).all()


def test_report_prints_the_whole_generated_section(pipeline):
    _, outputs = pipeline
    text = outputs["report"]
    assert text.startswith("### Headline")
    assert "| Signal | TriviaQA test AUROC |" in text
    assert "chosen on the validation split" in text
    assert "Shuffled labels" in text
    assert "### Did the probe beat the model's own confidence?" in text
    assert "### Controls" in text
    assert "Where the probe and the confidence signal disagree" in text
    assert text.count("\n|") >= 5


def test_report_can_fill_in_a_readme(pipeline, tmp_path):
    from typer.testing import CliRunner

    from answer_probe import cli, report

    root, _ = pipeline
    io.set_dirs(data=root / "data", results=root / "results")
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# t\n\n{report.BEGIN_MARKER}\nno run yet\n{report.END_MARKER}\n\ntail\n"
    )
    result = CliRunner().invoke(
        cli.app, ["report", "--write", "--readme", str(readme)]
    )
    assert result.exit_code == 0, result.output
    text = readme.read_text()
    assert "no run yet" not in text
    assert "| Signal | TriviaQA test AUROC |" in text
    assert text.startswith("# t") and text.rstrip().endswith("tail")


def test_label_command_writes_a_blind_grading_sheet(pipeline):
    root, outputs = pipeline
    sheet = root / "data" / "human_grading_sheet.jsonl"
    assert sheet.exists()
    rows = io.read_jsonl(sheet)
    assert len(rows) == 60
    for row in rows:
        assert row["human_correct"] is None
        assert set(row) == {
            "qid",
            "dataset",
            "question",
            "model_answer",
            "gold_answers",
            "human_correct",
        }
    assert "No human grades yet" in outputs["label"]


def test_grading_sample_is_stratified_by_automatic_label(pipeline):
    root, _ = pipeline
    labels = pd.read_csv(_results(root) / "labels.csv").set_index("qid")
    graded = [row["qid"] for row in io.read_jsonl(root / "data" / "human_grading_sheet.jsonl")]
    correct = labels.loc[graded, "correct"]
    assert correct.sum() == 30
    assert (1 - correct).sum() == 30


def test_reported_metrics_are_recomputable_from_the_scores_file(pipeline):
    """Every headline number must be derivable from a committed file.

    `metrics.csv` is recomputed here straight from `scores.csv` with the metric
    functions, independently of the code path `aprobe evaluate` took. If these
    ever disagree, a number in the README is not backed by the data in the repo.
    """
    from answer_probe.metrics import accuracy_at_coverage, auprc, auroc

    root, _ = pipeline
    scores = pd.read_csv(_results(root) / "scores.csv")
    metrics = pd.read_csv(_results(root) / "metrics.csv")

    checked = 0
    for _, row in metrics.iterrows():
        frame = scores[scores["split"] == row["split"]]
        y = frame["wrong"].to_numpy(dtype=int)
        s = frame[row["signal"]].to_numpy(dtype=float)
        assert row["auroc"] == pytest.approx(auroc(s, y), abs=1e-5)
        assert row["auprc"] == pytest.approx(auprc(s, y), abs=1e-5)
        assert row["acc_at_80"] == pytest.approx(accuracy_at_coverage(s, y, 0.8), abs=1e-5)
        assert row["acc_at_50"] == pytest.approx(accuracy_at_coverage(s, y, 0.5), abs=1e-5)
        assert row["error_rate"] == pytest.approx(y.mean(), abs=1e-4)
        checked += 1
    assert checked >= 18, "expected every signal on both splits"


def test_bootstrap_intervals_are_reproducible(pipeline):
    """Re-running evaluate must give the same intervals, not just the same points."""
    from typer.testing import CliRunner

    root, _ = pipeline
    first = pd.read_csv(_results(root) / "metrics.csv")
    result = CliRunner().invoke(
        cli.app, ["evaluate", "--run-dir", str(root / "run"), "--n-boot", "80"]
    )
    assert result.exit_code == 0, result.output
    again = pd.read_csv(_results(root) / "metrics.csv")
    pd.testing.assert_frame_equal(first, again)
