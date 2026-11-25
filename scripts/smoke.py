"""Run the whole analysis pipeline on a synthetic run, into a throwaway folder.

    python scripts/smoke.py /tmp/aprobe-smoke

Proves `label`, `train`, `evaluate` and `report` work end to end without a GPU
or a notebook run. The numbers it prints are not results: the activations are
random with a planted direction. Nothing it writes belongs in `results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from typer.testing import CliRunner  # noqa: E402

import fake_run  # noqa: E402
from answer_probe import cli, io  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aprobe-smoke").resolve()
    run_dir = root / "run"
    fake_run.write(run_dir, n_main=600, n_shift=250, seed=0)
    io.set_dirs(data=root / "data", results=root / "results")

    runner = CliRunner()
    for args in (
        ["label", "--run-dir", str(run_dir)],
        ["train", "--run-dir", str(run_dir)],
        ["evaluate", "--run-dir", str(run_dir), "--n-boot", "200"],
        ["report"],
    ):
        print(f"\n$ aprobe {' '.join(args)}")
        result = runner.invoke(cli.app, args)
        print(result.output, end="")
        if result.exit_code != 0:
            print(f"FAILED: {result.exception!r}")
            return 1
    print(f"\nSynthetic outputs under {root}. These are not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
