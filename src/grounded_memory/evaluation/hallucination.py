"""Hallucination and evidence-support rates over labeled claims.

Inputs are per-claim evidence labels (supported / contradicted / not_verifiable),
which callers produce with ``evaluation.grounding`` or human review. We report
``not_verifiable`` explicitly rather than collapsing it into either bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

from grounded_memory.evaluation.grounding import EvidenceLabel


@dataclass
class ClaimBreakdown:
    supported: int
    contradicted: int
    not_verifiable: int

    @property
    def total(self) -> int:
        return self.supported + self.contradicted + self.not_verifiable


def summarize_labels(labels: list[EvidenceLabel]) -> ClaimBreakdown:
    return ClaimBreakdown(
        supported=sum(1 for x in labels if x == EvidenceLabel.SUPPORTED),
        contradicted=sum(1 for x in labels if x == EvidenceLabel.CONTRADICTED),
        not_verifiable=sum(1 for x in labels if x == EvidenceLabel.NOT_VERIFIABLE),
    )


def hallucination_rate(labels: list[EvidenceLabel], strict: bool = True) -> float:
    """Unsupported-claim rate.

    - strict=True  (default): only CONTRADICTED claims count as hallucinations;
      denominator is verifiable claims (supported + contradicted). This avoids
      penalizing the model for annotation gaps.
    - strict=False: treat NOT_VERIFIABLE as unsupported too; denominator is all
      claims. Report both if you use this — it is an upper bound.
    """
    b = summarize_labels(labels)
    if strict:
        denom = b.supported + b.contradicted
        return b.contradicted / denom if denom else 0.0
    return (b.contradicted + b.not_verifiable) / b.total if b.total else 0.0


def evidence_support_rate(labels: list[EvidenceLabel]) -> float:
    """Fraction of claims that are supported, over verifiable claims."""
    b = summarize_labels(labels)
    denom = b.supported + b.contradicted
    return b.supported / denom if denom else 0.0


__all__ = [
    "ClaimBreakdown",
    "summarize_labels",
    "hallucination_rate",
    "evidence_support_rate",
]
