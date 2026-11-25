"""Answer normalization and gold-alias matching.

The base normalization is the standard SQuAD recipe: lowercase, drop
punctuation, drop articles, collapse whitespace. On top of that we canonicalise
number words to digits, because "nineteen sixty nine" and "1969" are the same
answer and a plain exact-match scorer calls one of them wrong.

Correct means either
  * the normalized prediction equals a normalized gold alias, or
  * a normalized gold alias appears as a whole-word span inside a *short*
    normalized prediction ("he was born in 1969" counts for "1969"; an essay
    that happens to contain the right word does not).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ARTICLES = frozenset({"a", "an", "the"})
# Apostrophes are deleted so "Dog's" and "Dogs" agree; every other punctuation
# mark becomes a space so "Jean-Luc" and "Jean Luc" agree. SQuAD's own recipe
# deletes both, which makes hyphenated aliases ("Jean-Luc" -> "jeanluc") fail to
# match their spaced variants, and TriviaQA's alias lists are inconsistent about
# hyphens.
_APOSTROPHE_RE = re.compile(r"[’'`´]", flags=re.UNICODE)
# Keep a period only when it sits between two digits, so decimals survive.
_PUNCT_RE = re.compile(r"[^\w\s.]|(?<!\d)\.|\.(?!\d)", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

# How many words a prediction may have and still count as a "short answer" for
# the whole-word span rule.
MAX_SPAN_WORDS = 10

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
    "hundredth": 100, "thousandth": 1000,
}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | set(_ORDINALS) | {"and"}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _collapse_number_run(words: list[str]) -> str | None:
    """Turn a run of number words into a digit string, or None if it isn't one.

    Handles cardinals up to billions ("two thousand and five" -> 2005), simple
    ordinals ("twenty first" -> 21), and the way years are spoken in pairs
    ("nineteen sixty nine" -> 1969).
    """
    if not words or words == ["and"]:
        return None
    if words[-1] == "and":  # a trailing "and" belongs to the surrounding text
        return None

    # Year-style pairs: two 2-digit numbers in a row, no scale word involved.
    if (
        len(words) in (2, 3)
        and not any(w in _SCALES for w in words)
        and words[0] in _UNITS
        and 10 <= _UNITS[words[0]] <= 19
    ):
        tail = [w for w in words[1:] if w != "and"]
        rest = _collapse_number_run(tail) if tail else None
        if rest is not None and 0 < int(rest) < 100:
            return f"{_UNITS[words[0]]}{int(rest):02d}"

    total = 0
    current = 0
    seen_any = False
    for word in words:
        if word == "and":
            continue
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word in _ORDINALS:
            current += _ORDINALS[word]
        elif word in _SCALES:
            scale = _SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        else:
            return None
        seen_any = True
    if not seen_any:
        return None
    return str(total + current)


def _canonicalise_numbers(words: list[str]) -> list[str]:
    """Replace runs of number words with their digit form."""
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] in _NUMBER_WORDS and words[i] != "and":
            j = i
            while j < len(words) and words[j] in _NUMBER_WORDS:
                j += 1
            run = words[i:j]
            # Give back any trailing "and" we swallowed.
            while run and run[-1] == "and":
                run.pop()
                j -= 1
            collapsed = _collapse_number_run(run)
            if collapsed is not None:
                out.append(collapsed)
            else:
                out.extend(run)
            i = j
            continue
        # "21st" / "3rd" -> "21" / "3"
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", words[i])
        out.append(m.group(1) if m else words[i])
        i += 1
    return out


def normalize_answer(text: str, *, numbers: bool = True) -> str:
    """SQuAD-style normalization, optionally canonicalising number words."""
    if text is None:
        return ""
    text = _strip_accents(str(text)).lower()
    # Commas inside numerals ("1,000") would otherwise split the numeral in two.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = _APOSTROPHE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    words = [w for w in _WS_RE.split(text) if w and w not in _ARTICLES]
    if numbers:
        words = _canonicalise_numbers(words)
    return " ".join(words)


@dataclass(frozen=True)
class Match:
    """Why an answer was marked correct (or not)."""

    correct: bool
    rule: str  # "exact" | "span" | "none"
    alias: str | None = None


def match_answer(
    prediction: str,
    aliases: list[str],
    *,
    max_span_words: int = MAX_SPAN_WORDS,
) -> Match:
    """Score one prediction against a question's gold aliases."""
    pred = normalize_answer(prediction)
    if not pred:
        return Match(False, "none")

    norm_aliases = [a for a in (normalize_answer(x) for x in aliases) if a]
    for alias in norm_aliases:
        if pred == alias:
            return Match(True, "exact", alias)

    if len(pred.split()) <= max_span_words:
        for alias in norm_aliases:
            if re.search(rf"(?<!\S){re.escape(alias)}(?!\S)", pred):
                return Match(True, "span", alias)

    return Match(False, "none")


def is_correct(prediction: str, aliases: list[str]) -> bool:
    """True if the prediction matches any gold alias."""
    return match_answer(prediction, aliases).correct


# Phrases that mean "I am not answering". The prompt asks for "unknown", but
# instruction-tuned models paraphrase. Stored normalized so the comparison in
# `is_refusal` uses exactly the same transformation as the phrases themselves.
_REFUSAL_PHRASES = (
    "unknown",
    "I don't know",
    "I do not know",
    "don't know",
    "no idea",
    "not known",
    "no answer",
    "none",
    "N/A",
    "unsure",
    "I am not sure",
    "I'm not sure",
    "not sure",
    "cannot determine",
    "I cannot answer",
    "unanswerable",
)
_REFUSALS = frozenset(normalize_answer(p, numbers=False) for p in _REFUSAL_PHRASES)


def is_refusal(prediction: str) -> bool:
    """True if the model declined to answer rather than committing to one."""
    pred = normalize_answer(prediction, numbers=False)
    if not pred:
        return True
    return pred in _REFUSALS
