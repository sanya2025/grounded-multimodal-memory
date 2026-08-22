"""Attribute accuracy (color, material, size, state, ...)."""

from __future__ import annotations

from grounded_memory.evaluation.categorical import tuple_prf1
from grounded_memory.evaluation.normalize import normalize_simple
from grounded_memory.evaluation.objects import PRF1


def attribute_accuracy(predicted: set[tuple[str, str]], reference: set[tuple[str, str]]) -> PRF1:
    """Score (object, attribute) pairs: precision/recall/F1 over reference pairs."""
    return tuple_prf1(predicted, reference, normalizer=normalize_simple)


__all__ = ["attribute_accuracy"]
