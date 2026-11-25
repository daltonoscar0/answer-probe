"""Hand-written cases for answer normalization and alias matching."""

import pytest

from answer_probe.normalize import (
    is_correct,
    is_refusal,
    match_answer,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Beatles", "beatles"),
        ("  a  Dog's   life! ", "dogs life"),
        ("Jean-Luc Picard", "jean luc picard"),
        ("Beyoncé", "beyonce"),
        ("AN apple.", "apple"),
        ("", ""),
        ("1,000", "1000"),
        ("$5.5 million", "5.5 1000000"),  # scale words collapse on both sides, so this matches
        ("Washington, D.C.", "washington d c"),
    ],
)
def test_squad_normalization(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("nineteen sixty nine", "1969"),
        ("Nineteen Eighty-Four", "1984"),
        ("two thousand and five", "2005"),
        ("twenty three", "23"),
        ("three hundred", "300"),
        ("two million", "2000000"),
        ("forty-two", "42"),
        ("the twenty first", "21"),
        ("21st", "21"),
        ("third", "3"),
        ("seven", "7"),
    ],
)
def test_number_words_become_digits(raw, expected):
    assert normalize_answer(raw) == expected


def test_number_canonicalisation_leaves_prose_alone():
    # "one" here is a pronoun, but canonicalising it to "1" is harmless and
    # consistent on both sides of the comparison. What must not happen is a
    # crash or a swallowed neighbouring word.
    assert normalize_answer("Lord of the Rings") == "lord of rings"
    assert normalize_answer("and then") == "and then"
    assert normalize_answer("one hundred and one dalmatians") == "101 dalmatians"


def test_exact_match():
    m = match_answer("The Beatles", ["Beatles", "The Beatles"])
    assert m.correct and m.rule == "exact"


def test_span_match_in_short_answer():
    m = match_answer("He was born in 1969.", ["1969"])
    assert m.correct and m.rule == "span" and m.alias == "1969"


def test_span_match_respects_word_boundaries():
    assert not is_correct("Washington State", ["ash"])
    assert not is_correct("Paris", ["is"])


def test_span_match_rejected_in_long_answer():
    long_answer = " ".join(["word"] * 12) + " 1969"
    assert not is_correct(long_answer, ["1969"])


def test_number_word_answer_matches_digit_alias():
    assert is_correct("nineteen sixty-nine", ["1969"])
    assert is_correct("1969", ["nineteen sixty nine"])


def test_date_with_qualifier():
    assert is_correct("in July 1969", ["July 1969"])
    assert is_correct("20 July 1969", ["July 20, 1969", "20 July 1969"])


def test_wrong_answer_is_wrong():
    assert not is_correct("Paris", ["London", "City of London"])


def test_empty_and_missing():
    assert not is_correct("", ["Paris"])
    assert not is_correct("   ", ["Paris"])
    assert not is_correct("Paris", [])


def test_alias_that_normalizes_to_nothing_is_ignored():
    assert not is_correct("Paris", ["the", "a"])


@pytest.mark.parametrize(
    "text", ["unknown", "Unknown.", "I don't know", "I do not know.", "", "   ", "N/A"]
)
def test_refusals(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", ["Paris", "1969", "The Beatles", "none of the above"])
def test_non_refusals(text):
    assert not is_refusal(text)
