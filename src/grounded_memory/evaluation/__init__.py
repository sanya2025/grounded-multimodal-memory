"""Reusable, deterministic evaluation metrics.

Design notes:
- Metric code is deterministic (no LLM judge as a primary metric).
- Scene graphs are treated as references, not omniscient ground truth: evidence
  labels are tri-state (supported / contradicted / not_verifiable), so an
  unannotated claim is NOT automatically a hallucination.
"""

from __future__ import annotations

from grounded_memory.evaluation.abstention import (
    abstention_metrics,
    is_abstention,
)
from grounded_memory.evaluation.categorical import accuracy
from grounded_memory.evaluation.hallucination import (
    evidence_support_rate,
    hallucination_rate,
)
from grounded_memory.evaluation.objects import object_prf1
from grounded_memory.evaluation.qa import normalize_answer, qa_accuracy
from grounded_memory.evaluation.retrieval import mrr, ndcg_at_k, recall_at_k
from grounded_memory.evaluation.stats import bootstrap_ci, paired_bootstrap

__all__ = [
    "object_prf1",
    "accuracy",
    "hallucination_rate",
    "evidence_support_rate",
    "is_abstention",
    "abstention_metrics",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "qa_accuracy",
    "normalize_answer",
    "bootstrap_ci",
    "paired_bootstrap",
]
