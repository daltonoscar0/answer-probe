"""Checkpointing has to survive a Colab disconnect, so it is tested like one."""

import ml_dtypes
import numpy as np
import pytest

from answer_probe.shards import Progress, ShardWriter, format_duration

BF16 = ml_dtypes.bfloat16

LAYERS, HIDDEN = 4, 6


def row(value):
    return np.full((LAYERS, HIDDEN), value, dtype=BF16)


def write_n(directory, start, stop, shard_size=10):
    writer = ShardWriter(directory, shard_size=shard_size)
    assert writer.resume_at() == start
    for i in range(start, stop):
        writer.add({"qid": f"q{i}", "answer": f"a{i}"}, row(i), row(-i))
    writer.close()
    return writer


def test_records_and_arrays_round_trip(tmp_path):
    write_n(tmp_path, 0, 25)
    records, before, after = ShardWriter(tmp_path, shard_size=10).read_all()
    assert [r["qid"] for r in records] == [f"q{i}" for i in range(25)]
    assert before.shape == (25, LAYERS, HIDDEN)
    assert before.dtype == BF16
    np.testing.assert_array_equal(before[7], row(7))
    np.testing.assert_array_equal(after[7], row(-7))


def test_resume_picks_up_after_the_last_complete_shard(tmp_path):
    write_n(tmp_path, 0, 20)
    resumed = ShardWriter(tmp_path, shard_size=10)
    assert resumed.resume_at() == 20
    for i in range(20, 35):
        resumed.add({"qid": f"q{i}"}, row(i), row(-i))
    resumed.close()
    records, before, _ = ShardWriter(tmp_path, shard_size=10).read_all()
    assert [r["qid"] for r in records] == [f"q{i}" for i in range(35)]
    assert before.shape == (35, LAYERS, HIDDEN)


def test_resume_counts_records_not_shards(tmp_path):
    """A short final shard must not make the next run skip questions."""
    write_n(tmp_path, 0, 13)  # one full shard of 10 plus a short one of 3
    writer = ShardWriter(tmp_path, shard_size=10)
    assert writer.committed_shards() == 2
    assert writer.resume_at() == 13
    for i in range(13, 18):
        writer.add({"qid": f"q{i}"}, row(i), row(-i))
    writer.close()
    records, before, _ = ShardWriter(tmp_path, shard_size=10).read_all()
    assert [r["qid"] for r in records] == [f"q{i}" for i in range(18)]
    assert len(before) == 18


def test_an_interrupted_write_is_not_counted(tmp_path):
    """A crash between the .npz and the .jsonl must not look like progress."""
    write_n(tmp_path, 0, 20)
    npz, jsonl = ShardWriter(tmp_path, shard_size=10).paths(1)
    jsonl.unlink()  # the commit marker never landed
    writer = ShardWriter(tmp_path, shard_size=10)
    assert writer.committed_shards() == 1
    assert writer.resume_at() == 10
    records, before, _ = writer.read_all()
    assert len(records) == 10 and len(before) == 10
    assert npz.exists()  # the orphan array file is simply overwritten


def test_rerunning_from_a_resume_point_overwrites_cleanly(tmp_path):
    write_n(tmp_path, 0, 20)
    (tmp_path / "shard_00001.jsonl").unlink()
    writer = ShardWriter(tmp_path, shard_size=10)
    for i in range(10, 20):
        writer.add({"qid": f"q{i}"}, row(i), row(-i))
    writer.close()
    records, before, _ = ShardWriter(tmp_path, shard_size=10).read_all()
    assert [r["qid"] for r in records] == [f"q{i}" for i in range(20)]
    np.testing.assert_array_equal(before[15], row(15))


def test_records_only_writer_needs_no_arrays(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=4, with_arrays=False)
    for i in range(10):
        writer.add({"qid": f"q{i}", "agreement": i / 10})
    writer.close()
    records, before, after = ShardWriter(tmp_path, shard_size=4, with_arrays=False).read_all()
    assert len(records) == 10
    assert before is None and after is None
    assert not list(tmp_path.glob("*.npz"))


def test_array_writer_rejects_a_record_without_activations(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=4)
    with pytest.raises(ValueError, match="activations"):
        writer.add({"qid": "q0"})


def test_empty_directory_reads_as_empty(tmp_path):
    records, before, after = ShardWriter(tmp_path, shard_size=10).read_all()
    assert records == [] and before is None and after is None


def test_shard_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="shard_size"):
        ShardWriter(tmp_path, shard_size=0)


def test_unicode_survives_the_round_trip(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=2, with_arrays=False)
    writer.add({"qid": "q0", "answer": "Beyoncé — “Déjà Vu”"})
    writer.close()
    records, _, _ = ShardWriter(tmp_path, shard_size=2, with_arrays=False).read_all()
    assert records[0]["answer"] == "Beyoncé — “Déjà Vu”"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5, "5s"), (89, "89s"), (120, "2 min"), (3600, "60 min"), (7200, "2.0 h")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_progress_estimates_the_time_left():
    clock = FakeClock()
    progress = Progress(1000, clock=clock)
    clock.now = 10.0
    progress.update(100)  # 100 questions in 10 seconds
    assert progress.rate() == pytest.approx(10.0)
    assert progress.remaining() == pytest.approx(90.0)
    # The line reports how far along we are, not how much is left.
    assert "100/1000" in progress.line()


def test_progress_accounts_for_work_done_in_an_earlier_session():
    clock = FakeClock()
    progress = Progress(1000, done_already=600, clock=clock)
    clock.now = 5.0
    progress.update(100)  # 100 in 5 seconds, 300 of the 1000 still to go
    assert progress.remaining() == pytest.approx(15.0)
    assert "700/1000" in progress.line()


def test_progress_is_patient_before_the_first_update():
    progress = Progress(1000, clock=FakeClock())
    assert progress.remaining() == float("inf")
    assert "left" in progress.line()


def test_unpack_rejects_float16_instead_of_misreading_it():
    """float16 is also two bytes; viewing its bits as bfloat16 would be silent."""
    from answer_probe.io import pack_bfloat16, unpack_bfloat16

    values = np.array([[1.0, 2.5, -3.75]], dtype=np.float32)
    packed = pack_bfloat16(values.astype(BF16))
    assert packed.dtype == np.uint16
    np.testing.assert_array_equal(
        unpack_bfloat16(packed).astype(np.float32), values.astype(BF16).astype(np.float32)
    )
    with pytest.raises(ValueError, match="not bfloat16 storage"):
        unpack_bfloat16(values.astype(np.float16))
    with pytest.raises(ValueError, match="not bfloat16 storage"):
        unpack_bfloat16(values.astype(np.float32))


def test_unpack_passes_bfloat16_through_unchanged():
    from answer_probe.io import unpack_bfloat16

    array = np.array([[1.0, 2.0]], dtype=np.float32).astype(BF16)
    assert unpack_bfloat16(array) is array
