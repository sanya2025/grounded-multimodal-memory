"""End-to-end QA accuracy with light answer normalization.

Kept separate from retrieval metrics on purpose::

    retrieval quality != answer quality

A system can retrieve the right event and still answer wrong, or guess correctly
despite poor retrieval. Analyze both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLES = {"a", "an", "the"}
_YES = {"yes", "yeah", "yep", "correct", "true"}
_NO = {"no", "nope", "false", "incorrect"}


def normalize_answer(answer: str) -> str:
    """Lower-case, strip punctuation/articles, collapse whitespace, map yes/no."""
    s = answer.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    s = " ".join(tokens)
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    return s


def answer_matches(prediction: str, reference: str) -> bool:
    """Match if normalized reference equals, or is contained in, the prediction."""
    p = normalize_answer(prediction)
    r = normalize_answer(reference)
    if not r:
        return False
    if p == r:
        return True
    # Allow the gold short answer to appear within a longer generated sentence.
    return bool(re.search(rf"\b{re.escape(r)}\b", p))


@dataclass
class QAResult:
    accuracy: float
    correct: int
    total: int


def qa_accuracy(predictions: list[str], references: list[str]) -> QAResult:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    total = len(predictions)
    correct = sum(1 for p, r in zip(predictions, references, strict=True) if answer_matches(p, r))
    return QAResult(correct / total if total else 0.0, correct, total)


__all__ = ["normalize_answer", "answer_matches", "QAResult", "qa_accuracy"]
