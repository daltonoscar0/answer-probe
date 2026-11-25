"""Paths, and reading the folder the notebook produced.

The activation arrays are hundreds of megabytes, so they are memory-mapped and
read one layer at a time. Nothing in the pipeline ever holds a whole
`[questions, layers, hidden]` tensor in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ml_dtypes
import numpy as np
import pandas as pd

# Activations are stored as bfloat16: two bytes per value like float16, but with
# float32's exponent range, so Gemma's large residual outliers are nowhere near
# the ceiling and nothing is clipped. It is also the dtype the model computes
# in, so the stored states are exactly the states the model had.
#
# The .npy format cannot record a dtype it does not know, so the bit patterns go
# to disk as uint16 and are viewed back on load. The true dtype is in the
# manifest, and uint16 keeps the files readable by numpy alone.
BF16 = ml_dtypes.bfloat16


def pack_bfloat16(array: np.ndarray) -> np.ndarray:
    """bfloat16 -> the uint16 bit patterns written to disk."""
    array = np.asarray(array)
    if array.dtype != BF16:
        array = array.astype(BF16)
    return array.view(np.uint16)


def unpack_bfloat16(array: np.ndarray) -> np.ndarray:
    """uint16 (or an unrecognised 2-byte dtype) read from disk -> bfloat16."""
    if array.dtype == BF16:
        return array
    if array.dtype.itemsize != 2:
        raise ValueError(f"expected a 2-byte dtype for bfloat16 storage, got {array.dtype}")
    return array.view(BF16)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Output locations. Functions rather than constants so tests (and anyone with
# the package installed outside a checkout) can point them elsewhere.
_DIRS = {"data": REPO_ROOT / "data", "results": REPO_ROOT / "results"}


def data_dir() -> Path:
    return _DIRS["data"]


def results_dir() -> Path:
    return _DIRS["results"]


def default_run_dir() -> Path:
    """Where the folder downloaded from the notebook is expected to be unzipped."""
    return data_dir() / "raw" / "run"


def set_dirs(*, data: str | Path | None = None, results: str | Path | None = None) -> None:
    if data is not None:
        _DIRS["data"] = Path(data)
    if results is not None:
        _DIRS["results"] = Path(results)

POSITIONS = ("before", "after")
POSITION_LABELS = {
    "before": "before answering (last prompt token)",
    "after": "after answering (last answer token)",
}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def index_by_qid(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a frame by question id, keeping `qid` as a column.

    The index name is cleared: pandas refuses to sort or merge on a label that
    is both an index level and a column.
    """
    frame = frame.set_index("qid", drop=False)
    frame.index.name = None
    return frame


class RunNotFound(FileNotFoundError):
    """Raised with a pointer to the notebook when the run folder is missing."""


@dataclass
class Run:
    """One extraction run: the questions, the answers, and the activations."""

    root: Path

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    @cached_property
    def questions(self) -> pd.DataFrame:
        return index_by_qid(pd.DataFrame(read_jsonl(self.root / "questions.jsonl")))

    @cached_property
    def generations(self) -> pd.DataFrame:
        return index_by_qid(pd.DataFrame(read_jsonl(self.root / "generations.jsonl")))

    @cached_property
    def selfconsistency(self) -> pd.DataFrame:
        path = self.root / "selfconsistency.jsonl"
        if not path.exists():
            return index_by_qid(pd.DataFrame(columns=["qid", "agreement", "n_samples"]))
        return index_by_qid(pd.DataFrame(read_jsonl(path)))

    @cached_property
    def activation_index(self) -> dict[str, int]:
        payload = json.loads((self.root / "activations" / "index.json").read_text())
        return {qid: i for i, qid in enumerate(payload["qids"])}

    @property
    def n_layers(self) -> int:
        return int(self.manifest["n_layers"])

    @property
    def hidden_size(self) -> int:
        return int(self.manifest["hidden_size"])

    def _memmap(self, position: str) -> np.ndarray:
        if position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
        path = self.root / "activations" / f"hidden_{position}.npy"
        if not path.exists():
            raise RunNotFound(f"missing {path}; run notebooks/extract.ipynb first")
        return unpack_bfloat16(np.load(path, mmap_mode="r"))

    def layer(self, position: str, layer: int, qids: list[str]) -> np.ndarray:
        """Hidden states at one layer for the given questions, as float32.

        Stored as bfloat16 and widened to float32 here: every probe, the
        standardiser and the metrics all run in float32, and widening is exact.

        Returns `[len(qids), hidden_size]` in the order `qids` was given.
        """
        array = self._memmap(position)
        index = self.activation_index
        try:
            rows = np.array([index[qid] for qid in qids], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover - guards a corrupt run
            raise KeyError(f"question {exc.args[0]!r} has no saved activations") from exc
        # Sort the row indices so the memory map is read in one forward sweep,
        # then put them back in the caller's order.
        order = np.argsort(rows, kind="mergesort")
        gathered = np.asarray(array[rows[order], layer, :], dtype=np.float32)
        out = np.empty_like(gathered)
        out[order] = gathered
        return out


def load_run(run_dir: str | Path | None = None) -> Run:
    root = Path(run_dir) if run_dir else default_run_dir()
    if not (root / "manifest.json").exists():
        raise RunNotFound(
            f"no extraction run at {root}\n"
            "Run notebooks/extract.ipynb on Colab, download the output folder, "
            f"and unzip it to {root} (or pass --run-dir)."
        )
    return Run(root)


def load_labels(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else results_dir() / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `aprobe label` first")
    return index_by_qid(pd.read_csv(path))


def write_run_folder(
    root: str | Path,
    *,
    manifest: dict,
    questions: list[dict],
    generations: list[dict],
    selfconsistency: list[dict],
    before: np.ndarray,
    after: np.ndarray,
    qids: list[str],
    data_summary: pd.DataFrame | None = None,
    split_map: dict[str, list[str]] | None = None,
) -> Path:
    """Write the folder the notebook hands over to the analysis pipeline.

    Both the notebook and the test fixture call this, so the schema the tests
    exercise is the schema the real run produces.
    """
    root = Path(root)
    (root / "activations").mkdir(parents=True, exist_ok=True)
    if before.shape != after.shape:
        raise ValueError(f"before {before.shape} and after {after.shape} disagree")
    if before.shape[0] != len(qids):
        raise ValueError(f"{before.shape[0]} activation rows for {len(qids)} question ids")

    np.save(root / "activations" / "hidden_before.npy", pack_bfloat16(before))
    np.save(root / "activations" / "hidden_after.npy", pack_bfloat16(after))
    write_json(root / "activations" / "index.json", {"qids": list(qids)})

    write_jsonl(root / "questions.jsonl", questions)
    write_jsonl(root / "generations.jsonl", generations)
    write_jsonl(root / "selfconsistency.jsonl", selfconsistency)
    if data_summary is not None:
        write_csv(root / "data_summary.csv", data_summary)
    if split_map is not None:
        from answer_probe.splits import save_splits

        save_splits(split_map, root / "splits.json")

    write_json(
        root / "manifest.json",
        {
            **manifest,
            "n_questions": len(qids),
            "n_layers": int(before.shape[1]),
            "hidden_size": int(before.shape[2]),
            "activation_dtype": str(np.dtype(BF16)),
            "activation_storage": (
                "uint16 bit patterns in the .npy files; view as bfloat16 "
                "(answer_probe.io.unpack_bfloat16)"
            ),
        },
    )
    return root
