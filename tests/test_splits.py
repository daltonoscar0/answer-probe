"""Splits are a partition, are reproducible, and keep the shift set out."""

import json

import pytest

from answer_probe.data import choose_indices
from answer_probe.splits import (
    SHIFT,
    TEST,
    TRAIN,
    VAL,
    check_splits,
    load_splits,
    make_splits,
    save_splits,
    selfconsistency_qids,
    split_of,
)


def corpus(n_main=200, n_shift=80):
    qids = [f"triviaqa:{i}" for i in range(n_main)] + [f"nq_open:{i}" for i in range(n_shift)]
    datasets = ["triviaqa"] * n_main + ["nq_open"] * n_shift
    return qids, datasets


def test_splits_are_a_partition():
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    assigned = [q for group in split.values() for q in group]
    assert sorted(assigned) == sorted(qids)
    assert len(assigned) == len(set(assigned))


def test_no_question_is_in_two_splits():
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    for a in split:
        for b in split:
            if a != b:
                assert not set(split[a]) & set(split[b])


def test_the_shift_set_is_entirely_held_out():
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    assert set(split[SHIFT]) == {q for q, d in zip(qids, datasets, strict=True) if d == "nq_open"}
    for name in (TRAIN, VAL, TEST):
        assert not any(q.startswith("nq_open:") for q in split[name])


def test_split_sizes_follow_the_fractions():
    qids, datasets = corpus(n_main=1000, n_shift=50)
    split = make_splits(qids, datasets)
    assert len(split[TRAIN]) == 700
    assert len(split[VAL]) == 150
    assert len(split[TEST]) == 150
    assert len(split[SHIFT]) == 50


def test_splits_are_reproducible_and_order_independent():
    qids, datasets = corpus()
    first = make_splits(qids, datasets)
    again = make_splits(qids, datasets)
    assert first == again
    # Reading the questions in a different order must not move anything.
    pairs = list(zip(qids, datasets, strict=True))[::-1]
    shuffled = make_splits([q for q, _ in pairs], [d for _, d in pairs])
    assert shuffled == first


def test_a_different_seed_gives_a_different_split():
    qids, datasets = corpus()
    assert make_splits(qids, datasets, seed=1) != make_splits(qids, datasets, seed=2)


def test_selfconsistency_covers_val_test_and_shift_but_not_train():
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    wanted = set(selfconsistency_qids(split))
    assert wanted == set(split[VAL]) | set(split[TEST]) | set(split[SHIFT])
    assert not wanted & set(split[TRAIN])


def test_check_splits_rejects_an_overlap():
    with pytest.raises(ValueError, match="shares"):
        check_splits({TRAIN: ["a", "b"], TEST: ["b", "c"]})


def test_check_splits_rejects_a_duplicate_within_a_split():
    with pytest.raises(ValueError, match="duplicate"):
        check_splits({TRAIN: ["a", "a"]})


def test_make_splits_rejects_duplicate_question_ids():
    with pytest.raises(ValueError, match="duplicate"):
        make_splits(["a", "a"], ["triviaqa", "triviaqa"])


def test_round_trip_through_json(tmp_path):
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    path = tmp_path / "splits.json"
    save_splits(split, path)
    assert load_splits(path) == split
    payload = json.loads(path.read_text())
    assert payload["counts"][TRAIN] == len(split[TRAIN])
    assert payload["seed"]


def test_split_of_inverts_the_map():
    qids, datasets = corpus()
    split = make_splits(qids, datasets)
    lookup = split_of(split)
    assert len(lookup) == len(qids)
    for name, group in split.items():
        for qid in group:
            assert lookup[qid] == name


def test_question_sampling_is_reproducible_and_sorted():
    first = choose_indices(17944, 6000, seed=20260101)
    again = choose_indices(17944, 6000, seed=20260101)
    assert list(first) == list(again)
    assert len(set(first.tolist())) == 6000
    assert list(first) == sorted(first)
    assert first.max() < 17944


def test_question_sampling_refuses_to_oversample():
    with pytest.raises(ValueError, match="asked for"):
        choose_indices(10, 20, seed=0)
