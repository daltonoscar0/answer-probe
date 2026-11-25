"""Question-surface features, answer-type guessing, and the confidence baselines.

The surface baseline exists to answer an obvious objection: maybe a probe only
picks up "this is a hard-looking question" rather than anything about the
model's state. It is fitted on features of the question text alone.
"""

from __future__ import annotations

import re

import numpy as np

from answer_probe.normalize import normalize_answer

# Checked in order, so "how many" wins over "how" and "what year" over "what".
_ANSWER_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("number", r"\bhow (?:many|much|old|long|tall|far|deep|fast)\b"),
    ("date", r"\b(?:when|what year|which year|what date|in what year|what decade)\b"),
    ("person", r"\b(?:who|whose|whom)\b"),
    ("place", r"\b(?:where|what city|which city|what country|which country)\b"),
    ("entity", r"\bwhich\b"),
    ("thing", r"\bwhat\b"),
    ("reason", r"\bwhy\b"),
    ("manner", r"\bhow\b"),
    ("command", r"^\s*(?:name|give|list|identify)\b"),
)

ANSWER_TYPES: tuple[str, ...] = (
    "number",
    "date",
    "person",
    "place",
    "entity",
    "thing",
    "reason",
    "manner",
    "command",
    "other",
)

_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


def answer_type(question: str) -> str:
    """Guess what kind of answer a question wants, from its question word.

    A crude heuristic, used only as a feature and as a way to group errors in
    the analysis -- never as a label.
    """
    text = question.lower()
    for name, pattern in _ANSWER_TYPE_PATTERNS:
        if re.search(pattern, text):
            return name
    return "other"


def gold_is_number(aliases) -> bool:
    """True if any gold alias is a bare number once normalized."""
    for alias in aliases:
        normalized = normalize_answer(alias)
        if normalized and all(_NUMERIC_RE.match(w) for w in normalized.split()):
            return True
    return False


SURFACE_FEATURE_NAMES: tuple[str, ...] = (
    "n_words",
    "n_chars",
    "has_digit",
    "gold_is_number",
    *(f"type_{t}" for t in ANSWER_TYPES),
)


def surface_features(questions: list[str], aliases: list[list[str]]) -> np.ndarray:
    """The question-surface design matrix, `[n_questions, n_features]`."""
    if len(questions) != len(aliases):
        raise ValueError("questions and aliases must be the same length")
    rows = []
    for question, alias_list in zip(questions, aliases, strict=True):
        kind = answer_type(question)
        rows.append(
            [
                float(len(question.split())),
                float(len(question)),
                float(bool(re.search(r"\d", question))),
                float(gold_is_number(alias_list)),
                *(1.0 if kind == t else 0.0 for t in ANSWER_TYPES),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


# Confidence signals saved by the notebook. `negate` says whether the raw value
# has to be flipped so that larger means "more likely wrong".
CONFIDENCE_SIGNALS: dict[str, dict] = {
    "first_token_prob": {
        "column": "first_token_prob",
        "negate": True,
        "label": "First-token probability",
    },
    "answer_logprob": {
        "column": "answer_logprob_mean",
        "negate": True,
        "label": "Answer log-prob (length-normalised)",
    },
    "first_token_entropy": {
        "column": "first_token_entropy",
        "negate": False,
        "label": "First-token entropy",
    },
    "selfconsistency": {
        "column": "selfconsistency_agreement",
        "negate": True,
        "label": "Self-consistency (8 samples, T=1.0)",
    },
}


def confidence_score(frame, name: str) -> np.ndarray:
    """Pull one confidence signal out of a labels frame, oriented so high = wrong."""
    spec = CONFIDENCE_SIGNALS[name]
    values = np.asarray(frame[spec["column"]], dtype=np.float64)
    return -values if spec["negate"] else values
