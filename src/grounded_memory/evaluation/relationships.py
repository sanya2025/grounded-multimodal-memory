"""Relationship accuracy over (subject, relation, object) triples."""

from __future__ import annotations

from grounded_memory.evaluation.categorical import AccuracyResult, tuple_accuracy
from grounded_memory.evaluation.normalize import normalize_object, normalize_relation


def relationship_accuracy(
    predicted: set[tuple[str, str, str]], reference: set[tuple[str, str, str]]
) -> AccuracyResult:
    """Fraction of reference relation triples recovered (after normalization)."""
    def norm_triple(t: tuple[str, str, str]) -> tuple[str, str, str]:
        s, r, o = t
        return (normalize_object(s), normalize_relation(r), normalize_object(o))

    pred = {norm_triple(t) for t in predicted}
    ref = {norm_triple(t) for t in reference}
    if not ref:
        return AccuracyResult(0.0, 0, 0)
    correct = len(pred & ref)
    return AccuracyResult(correct / len(ref), correct, len(ref))


__all__ = ["relationship_accuracy"]
