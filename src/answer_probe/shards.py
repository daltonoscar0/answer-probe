"""Resumable checkpoints for the extraction run.

Colab disconnects. The notebook writes a shard every few hundred questions to
Google Drive and, on the next run, picks up from the last complete one.

A shard is two files: an `.npz` holding the activations and a `.jsonl` holding
the records. The `.npz` is written first and the `.jsonl` is the commit marker,
so a session killed halfway through writing never leaves a shard that looks
complete. Shards are contiguous and in question order, so "how far did we get"
is just the number of complete shards times the shard size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ShardWriter:
    """Append records (and optionally activation rows), checkpointing in blocks."""

    def __init__(self, directory: str | Path, *, shard_size: int, with_arrays: bool = True):
        if shard_size < 1:
            raise ValueError("shard_size must be at least 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.with_arrays = with_arrays
        self._records: list[dict] = []
        self._before: list[np.ndarray] = []
        self._after: list[np.ndarray] = []
        self._next_index = self.committed_shards()

    def paths(self, index: int) -> tuple[Path, Path]:
        stem = f"shard_{index:05d}"
        return self.directory / f"{stem}.npz", self.directory / f"{stem}.jsonl"

    def committed_shards(self) -> int:
        """How many complete shards are on disk, counting from zero."""
        index = 0
        while True:
            npz, jsonl = self.paths(index)
            if not jsonl.exists() or (self.with_arrays and not npz.exists()):
                return index
            index += 1

    def resume_at(self) -> int:
        """The index of the first question not yet checkpointed.

        Counted from the records themselves rather than assuming every shard is
        full: a pass whose length is not a multiple of the shard size ends with
        a short shard, and multiplying out would skip the questions it is
        missing.
        """
        total = 0
        for index in range(self.committed_shards()):
            _, jsonl = self.paths(index)
            total += sum(1 for line in jsonl.read_text().splitlines() if line.strip())
        return total

    def add(
        self,
        record: dict,
        before: np.ndarray | None = None,
        after: np.ndarray | None = None,
    ) -> None:
        if self.with_arrays and (before is None or after is None):
            raise ValueError("this writer expects activations with every record")
        self._records.append(record)
        if self.with_arrays:
            self._before.append(np.asarray(before))
            self._after.append(np.asarray(after))
        if len(self._records) >= self.shard_size:
            self._write(self.shard_size)

    def close(self) -> int:
        """Write whatever is left. Returns the number of shards on disk."""
        while self._records:
            self._write(min(self.shard_size, len(self._records)))
        return self.committed_shards()

    def _write(self, count: int) -> None:
        npz, jsonl = self.paths(self._next_index)
        if self.with_arrays:
            np.savez(
                npz,
                before=np.stack(self._before[:count]),
                after=np.stack(self._after[:count]),
            )
            self._before = self._before[count:]
            self._after = self._after[count:]
        # The record file is the commit marker, so it goes last.
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in self._records[:count])
        jsonl.write_text(text + "\n")
        self._records = self._records[count:]
        self._next_index += 1

    def read_all(self) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
        """Every committed shard, concatenated in order."""
        records: list[dict] = []
        before: list[np.ndarray] = []
        after: list[np.ndarray] = []
        for index in range(self.committed_shards()):
            npz, jsonl = self.paths(index)
            records.extend(
                json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
            )
            if self.with_arrays:
                with np.load(npz) as payload:
                    before.append(payload["before"])
                    after.append(payload["after"])
        if not self.with_arrays:
            return records, None, None
        if not before:
            return records, None, None
        return records, np.concatenate(before), np.concatenate(after)


def format_duration(seconds: float) -> str:
    """A duration a person can read at a glance."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


class Progress:
    """Rate and time-remaining reporting for a long loop."""

    def __init__(self, total: int, *, done_already: int = 0, clock=None):
        import time

        self.total = total
        self.done_already = done_already
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self.processed = 0

    def update(self, processed: int) -> None:
        self.processed = processed

    def elapsed(self) -> float:
        return self._clock() - self._started

    def rate(self) -> float:
        elapsed = self.elapsed()
        return self.processed / elapsed if elapsed > 0 else 0.0

    def remaining(self) -> float:
        rate = self.rate()
        if rate <= 0:
            return float("inf")
        return (self.total - self.done_already - self.processed) / rate

    def line(self, label: str = "") -> str:
        position = self.done_already + self.processed
        return (
            f"  {position}/{self.total}  {self.rate():.1f} questions/s  "
            f"about {format_duration(self.remaining())} left{label}"
        )
