"""`aprobe`: label, train, evaluate, report.

Every command writes to `results/`, and every number in the README comes from
one of those files.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import typer

from answer_probe import analysis, evaluate, io, labeling, plots, report, splits, train

app = typer.Typer(add_completion=False, help=__doc__)

RUN_DIR_OPTION = typer.Option(None, "--run-dir", help="Folder downloaded from the notebook.")


def _echo(message: str) -> None:
    typer.echo(message)


def _load_splits(run: io.Run) -> dict[str, list[str]]:
    """Rebuild the splits from the question list, and check them against the run."""
    questions = run.questions
    built = splits.make_splits(list(questions["qid"]), list(questions["dataset"]))
    run_splits = run.root / "splits.json"
    if run_splits.exists():
        notebook_splits = splits.load_splits(run_splits)
        for name, qids in built.items():
            if sorted(qids) != sorted(notebook_splits.get(name, [])):
                raise typer.BadParameter(
                    f"split {name!r} differs from the notebook's splits.json; "
                    "the question sample or the split seed changed"
                )
    return built


@app.command()
def label(
    run_dir: Path = RUN_DIR_OPTION,
    grading_sheet: bool = typer.Option(
        True, help="Also write the blind grading sheet for the label audit."
    ),
) -> None:
    """Label the notebook's answers for correctness and audit those labels."""
    run = io.load_run(run_dir)
    split_map = _load_splits(run)
    splits.save_splits(split_map, io.results_dir() / "splits.json")

    frame = labeling.build_labels(run, split_map)
    io.write_csv(io.results_dir() / "labels.csv", frame.drop(columns=[]))
    accuracy = labeling.accuracy_table(frame)
    io.write_csv(io.results_dir() / "accuracy.csv", accuracy)

    summary = run.root / "data_summary.csv"
    if summary.exists():
        shutil.copyfile(summary, io.results_dir() / "data_summary.csv")
    io.write_json(io.results_dir() / "run_manifest.json", run.manifest)

    _echo(f"Labelled {len(frame)} questions -> results/labels.csv")
    _echo(report.accuracy_sentence(accuracy))
    for _, row in accuracy[accuracy["split"] == "all"].iterrows():
        _echo(
            f"  {row['dataset']}: n={row['n']} accuracy={row['accuracy']:.4f} "
            f"exact={row['exact_match_share']:.3f} span={row['span_match_share']:.3f}"
        )

    if grading_sheet:
        sample = labeling.grading_sample(frame)
        path = labeling.write_grading_sheet(sample, run.questions)
        _echo(f"Wrote {len(sample)} answers for blind grading -> {path}")
        _echo("Grade them with `aprobe grade`, then re-run `aprobe label`.")

    human = labeling.load_human_labels()
    agreement = labeling.agreement_report(frame, human)
    io.write_json(io.results_dir() / "label_agreement.json", agreement)
    if agreement["n_graded"]:
        _echo(
            f"Human agreement on {agreement['n_graded']} graded answers: "
            f"{agreement['agreement']:.1%} "
            f"({'meets' if agreement['meets_threshold'] else 'BELOW'} the "
            f"{labeling.AGREEMENT_THRESHOLD:.0%} threshold)"
        )
        if not agreement["meets_threshold"]:
            _echo("Fix the matcher in answer_probe/normalize.py and grade again.")
    else:
        _echo("No human grades yet: agreement not reported.")


@app.command()
def grade(limit: int = typer.Option(0, help="Stop after this many answers (0 = all).")) -> None:
    """Grade the sampled answers by hand, without seeing the automatic label."""
    sheet_path = labeling.grading_sheet_path()
    if not sheet_path.exists():
        raise typer.BadParameter(f"{sheet_path} not found; run `aprobe label` first")
    sheet = io.read_jsonl(sheet_path)
    done = {row["qid"] for row in labeling.load_human_labels().to_dict("records")}
    todo = [row for row in sheet if row["qid"] not in done]
    if not todo:
        _echo(f"All {len(sheet)} sampled answers are already graded.")
        return

    _echo(f"{len(todo)} of {len(sheet)} answers left. y = correct, n = wrong, q = stop.\n")
    graded = []
    for i, row in enumerate(todo, start=1):
        if limit and i > limit:
            break
        _echo(f"[{i}/{len(todo)}] {row['question']}")
        _echo(f"  model answer: {row['model_answer']!r}")
        _echo(f"  gold answers: {', '.join(map(str, row['gold_answers'][:8]))}")
        reply = typer.prompt("  correct? [y/n/q]", default="").strip().lower()
        if reply.startswith("q"):
            break
        if reply.startswith("y"):
            graded.append({**row, "human_correct": 1})
        elif reply.startswith("n"):
            graded.append({**row, "human_correct": 0})
        else:
            _echo("  skipped")
        _echo("")

    if graded:
        existing = (
            io.read_jsonl(labeling.human_labels_path())
            if labeling.human_labels_path().exists()
            else []
        )
        io.write_jsonl(labeling.human_labels_path(), existing + graded)
        _echo(f"Saved {len(graded)} grades -> {labeling.human_labels_path()}")
    _echo("Re-run `aprobe label` to see the agreement.")


@app.command("train")
def train_command(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Fit every probe at every layer, plus all controls and baselines."""
    run = io.load_run(run_dir)
    labels = io.load_labels()
    split_map = splits.load_splits(io.results_dir() / "splits.json")

    started = time.time()

    def progress(done, total, position, layer):
        elapsed = time.time() - started
        remaining = elapsed / done * (total - done)
        _echo(
            f"  [{done:>3}/{total}] {position} layer {layer:>2}  "
            f"elapsed {elapsed / 60:.1f} min, about {remaining / 60:.1f} min left"
        )

    _echo(f"Fitting probes at {run.n_layers} layers x {len(io.POSITIONS)} positions")
    artefacts = train.run_training(run, labels, split_map, progress=progress)

    io.write_csv(io.results_dir() / "layer_scan.csv", artefacts["scan"].round(5))
    io.write_csv(io.results_dir() / "controls.csv", artefacts["controls"].round(5))
    io.write_csv(io.results_dir() / "scores.csv", artefacts["scores"].round(6))
    io.write_json(
        io.results_dir() / "selection.json",
        {
            "selection": artefacts["selection"],
            "best_confidence_signal": artefacts["best_confidence"],
            "confidence_signals_available": artefacts["confidence_names"],
            "surface_baseline": {
                "l2": artefacts["surface"]["l2"],
                "weights": artefacts["surface"]["weights"],
            },
            "combination": {
                "l2": artefacts["combination"]["l2"],
                "weights": artefacts["combination"]["weights"],
                "confidence_signal": artefacts["combination"]["confidence_signal"],
            },
        },
    )
    headline = artefacts["selection"]["headline"]
    _echo(
        f"Headline probe: {headline['position']} answering, layer {headline['layer']}, "
        f"L2 {headline['l2']}, validation AUROC {headline['auroc_val']:.4f} "
        "(layer chosen on validation)"
    )
    _echo(report.controls_sentence(artefacts["controls"]))
    _echo(f"Best confidence baseline on validation: {artefacts['best_confidence']}")


@app.command("evaluate")
def evaluate_command(
    run_dir: Path = RUN_DIR_OPTION,
    n_boot: int = typer.Option(1000, help="Bootstrap resamples per interval."),
) -> None:
    """Compute every reported metric and write the plots to results/."""
    scores = pd.read_csv(io.results_dir() / "scores.csv")
    labels = io.load_labels()
    selection = json.loads((io.results_dir() / "selection.json").read_text())
    scan = pd.read_csv(io.results_dir() / "layer_scan.csv")
    run = io.load_run(run_dir)

    headline_signal = "probe_before_linear"
    metrics = evaluate.metrics_table(scores, n_boot=n_boot)
    io.write_csv(io.results_dir() / "metrics.csv", metrics.round(5))

    comparisons = evaluate.comparison_table(scores, headline_signal, n_boot=n_boot)
    io.write_csv(io.results_dir() / "comparisons.csv", comparisons.round(5))

    best_conf = evaluate.best_baseline_column(metrics)
    curve_columns = [headline_signal, best_conf, "combination"]
    rc = evaluate.risk_coverage_table(scores, curve_columns)
    io.write_csv(io.results_dir() / "risk_coverage.csv", rc)

    bins, calib = evaluate.calibration_table(scores, f"prob_{headline_signal[len('probe_'):]}")
    io.write_csv(io.results_dir() / "calibration_bins.csv", bins.round(5))
    io.write_csv(io.results_dir() / "calibration.csv", calib)

    committed = evaluate.refusal_breakdown(scores, curve_columns, n_boot=n_boot)
    io.write_csv(io.results_dir() / "excluding_refusals.csv", committed.round(5))

    io.write_csv(io.results_dir() / "error_concentration.csv", analysis.error_concentration(labels))

    for split in evaluate.REPORT_SPLITS:
        probe_caught, conf_caught = analysis.disagreements(
            scores, labels, probe_column=headline_signal, confidence_column=best_conf, split=split
        )
        for name, frame in (("probe_caught", probe_caught), ("confidence_caught", conf_caught)):
            if frame.empty:
                continue
            frame = frame.copy()
            frame["gold_answer"] = [
                analysis.gold_answers(labels, run.questions, [qid])[qid] for qid in frame["qid"]
            ]
            io.write_csv(io.results_dir() / f"examples_{name}_{split}.csv", frame)

    baseline_auroc = float(
        metrics[(metrics["split"] == "test") & (metrics["signal"] == best_conf)]["auroc"].iloc[0]
    )
    plots.layer_curve(
        scan,
        selection["selection"],
        baseline_auroc=baseline_auroc,
        baseline_label=evaluate.signal_label(best_conf),
        path=io.results_dir() / "layer_curve.png",
    )
    plots.risk_coverage_plot(rc, curve_columns, path=io.results_dir() / "risk_coverage.png")
    plots.calibration_plot(bins, calib, path=io.results_dir() / "calibration.png")

    _echo(f"Wrote metrics and three figures to {io.results_dir()}")
    _echo(report.comparison_sentence(comparisons, split="test", other=best_conf))
    _echo(report.comparison_sentence(comparisons, split="shift", other=best_conf))


@app.command("report")
def report_command(
    write: bool = typer.Option(
        False, "--write", help="Splice the results into README.md as well as printing them."
    ),
    readme: Path = typer.Option(None, help="README to update (default: the repo's)."),
) -> None:
    """Print the headline table as Markdown, and optionally update the README.

    Everything printed here is read from `results/`. Nothing in the README's
    generated block is ever typed by hand, so the two cannot disagree.
    """
    results = io.results_dir()
    metrics = pd.read_csv(results / "metrics.csv")
    comparisons = pd.read_csv(results / "comparisons.csv")
    controls = pd.read_csv(results / "controls.csv")
    accuracy = pd.read_csv(results / "accuracy.csv")
    calibration = pd.read_csv(results / "calibration.csv")
    excluding = pd.read_csv(results / "excluding_refusals.csv")
    selection = json.loads((results / "selection.json").read_text())
    agreement_path = results / "label_agreement.json"
    agreement = json.loads(agreement_path.read_text()) if agreement_path.exists() else None

    examples = {}
    for key in ("probe_caught", "confidence_caught"):
        path = results / f"examples_{key}_test.csv"
        if path.exists():
            examples[key] = pd.read_csv(path)

    section = report.render_results_section(
        metrics=metrics,
        comparisons=comparisons,
        controls=controls,
        accuracy=accuracy,
        selection=selection,
        calibration=calibration,
        excluding_refusals=excluding,
        agreement=agreement,
        examples=examples,
    )
    _echo(section)

    if write:
        target = Path(readme) if readme else io.REPO_ROOT / "README.md"
        changed = report.splice_into_readme(target, section)
        _echo(f"{'Updated' if changed else 'No change to'} {target}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
