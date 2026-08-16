"""Attribute accuracy (color, material, size, state, ...)."""

from __future__ import annotations

from grounded_memory.evaluation.categorical import AccuracyResult, tuple_accuracy
from grounded_memory.evaluation.normalize import normalize_simple


def attribute_accuracy(
    predicted: set[tuple[str, str]], reference: set[tuple[str, str]]
) -> AccuracyResult:
    """Score (object, attribute) pairs. Fraction of reference pairs recovered."""
    return tuple_accuracy(predicted, reference, normalizer=normalize_simple)


__all__ = ["attribute_accuracy"]
