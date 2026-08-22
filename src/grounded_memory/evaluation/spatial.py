"""Spatial-relation accuracy (left_of, above, on, ...)."""

from __future__ import annotations

from grounded_memory.evaluation.normalize import normalize_object, normalize_relation
from grounded_memory.evaluation.objects import PRF1


def spatial_accuracy(
    predicted: set[tuple[str, str, str]], reference: set[tuple[str, str, str]]
) -> PRF1:
    """Precision/recall/F1 over spatial (subject, relation, object) triples.

    Relation phrases go through the same normalize_relation() vocabulary as
    relationship_accuracy (see configs/evaluation.yaml 'relation_normalization'
    and 'spatial_relations.canonical'), not just underscore-replacement, so
    e.g. GQA's "in front of" and a model's "is in front of" both normalize to
    "in_front_of" instead of only matching on exact literal phrasing.
    """
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


__all__ = ["spatial_accuracy"]
