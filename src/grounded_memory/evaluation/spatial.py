"""Spatial-relation accuracy (left_of, above, on, ...)."""

from __future__ import annotations

from grounded_memory.evaluation.categorical import AccuracyResult, tuple_accuracy
from grounded_memory.evaluation.normalize import normalize_object


def spatial_accuracy(
    predicted: set[tuple[str, str, str]], reference: set[tuple[str, str, str]]
) -> AccuracyResult:
    """Fraction of reference spatial triples recovered.

    Spatial relations are assumed already canonical (see configs/evaluation.yaml
    'spatial_relations.canonical'); only subject/object labels are normalized.
    """
    def norm_triple(t: tuple[str, str, str]) -> tuple[str, str, str]:
        s, r, o = t
        return (normalize_object(s), r.strip().lower().replace(" ", "_"), normalize_object(o))

    pred = {norm_triple(t) for t in predicted}
    ref = {norm_triple(t) for t in reference}
    if not ref:
        return AccuracyResult(0.0, 0, 0)
    correct = len(pred & ref)
    return AccuracyResult(correct / len(ref), correct, len(ref))


__all__ = ["spatial_accuracy"]
