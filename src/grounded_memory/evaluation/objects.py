"""Object-level precision / recall / F1 against reference object sets."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_memory.evaluation.normalize import normalize_object


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def object_prf1(
    predicted: list[str], reference: list[str], normalize: bool = True
) -> PRF1:
    """Set-based precision/recall/F1 over object labels.

    - Precision = correct mentioned objects / all mentioned objects
    - Recall    = correctly identified reference objects / reference objects
    """
    if normalize:
        pred = {normalize_object(x) for x in predicted}
        ref = {normalize_object(x) for x in reference}
    else:
        pred = {x.lower() for x in predicted}
        ref = {x.lower() for x in reference}

    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF1(precision, recall, f1, tp, fp, fn)


__all__ = ["PRF1", "object_prf1"]
