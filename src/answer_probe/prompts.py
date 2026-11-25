"""The one prompt used for every question, and the Gemma 3 chat template.

This file is the single source of truth for the prompt. The extraction notebook
imports it, so the string in the repo is the string the model saw.
"""

from __future__ import annotations

# Short-answer instruction. Fixed for every question in both datasets.
#
# "unknown" is offered deliberately: an instruction-tuned model that can decline
# gives us a refusal category to analyse, and refusing is a different failure
# from answering confidently and wrongly.
INSTRUCTION = (
    "Answer the question with the shortest possible answer: a name, a date, a "
    "number, or a short phrase. Give the answer only, with no explanation and "
    'no full sentence. If you do not know the answer, reply exactly "unknown".'
)

# Gemma 3 instruction-tuned chat template. The model turn is left open so the
# next token the model produces is the first token of its answer. BOS is added
# by the tokenizer (`encode(..., add_bos=True)`), not here.
TEMPLATE = (
    "<start_of_turn>user\n"
    "{instruction}\n\nQuestion: {question}"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

PROMPT_VERSION = "v1"


def build_prompt(question: str) -> str:
    """Render the full prompt string for one question."""
    return TEMPLATE.format(instruction=INSTRUCTION, question=question.strip())
