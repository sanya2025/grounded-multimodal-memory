"""Relationship accuracy over (subject, relation, object) triples."""

from __future__ import annotations

from grounded_memory.evaluation.normalize import normalize_object, normalize_relation
from grounded_memory.evaluation.objects import PRF1


def relationship_accuracy(
    predicted: set[tuple[str, str, str]], reference: set[tuple[str, str, str]]
) -> PRF1:
    """Precision/recall/F1 over (subject, relation, object) triples, after normalization."""
    def norm_triple(t: tuple[str, str, str]) -> tuple[str, str, str]:
        s, r, o = t
        return (normalize_object(s), normalize_relation(r), normalize_object(o))

    pred = {norm_triple(t) for t in predicted}
    ref = {norm_triple(t) for t in reference}
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF1(precision, recall, f1, tp, fp, fn)


__all__ = ["relationship_accuracy"]
