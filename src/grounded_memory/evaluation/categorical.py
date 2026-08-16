"""Shared accuracy helper for attribute / relation / spatial evaluation.

Attributes, relationships, and spatial relations are all scored as matched
predictions against references after normalization. The dedicated modules
(attributes.py, relationships.py, spatial.py) wrap this with the right
normalizer so the experiment code and notebooks import clearly-named functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class AccuracyResult:
    accuracy: float
    correct: int
    total: int


def accuracy(
    predictions: Iterable[bool],
) -> AccuracyResult:
    """Accuracy from an iterable of per-item correctness booleans."""
    items = list(predictions)
    total = len(items)
    correct = sum(1 for x in items if x)
    return AccuracyResult(correct / total if total else 0.0, correct, total)


def tuple_accuracy(
    predicted: set[tuple[str, ...]],
    reference: set[tuple[str, ...]],
    normalizer: Callable[[str], str] | None = None,
) -> AccuracyResult:
    """Fraction of reference tuples recovered by the prediction set.

    Used for (subject, relation, object) triples and (object, attribute) pairs.
    """
    def norm(t: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalizer(x) for x in t) if normalizer else tuple(x.lower() for x in t)

    pred = {norm(t) for t in predicted}
    ref = {norm(t) for t in reference}
    if not ref:
        return AccuracyResult(0.0, 0, 0)
    correct = len(pred & ref)
    return AccuracyResult(correct / len(ref), correct, len(ref))


__all__ = ["AccuracyResult", "accuracy", "tuple_accuracy"]
