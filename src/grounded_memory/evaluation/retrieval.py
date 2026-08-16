"""Retrieval metrics: Recall@K, MRR, nDCG@K.

Each query has a set of relevant (ground-truth) event ids and a ranked list of
retrieved event ids. These are standard, deterministic implementations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved.

    Denominator is the number of relevant items (capped at k is NOT applied here;
    this is standard recall@k = |relevant ∩ top_k| / |relevant|).
    """
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mrr(retrieved_lists: Sequence[Sequence[str]], relevant_sets: Sequence[set[str]]) -> float:
    """Mean reciprocal rank across queries."""
    if not retrieved_lists:
        return 0.0
    rr = [reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_sets)]
    return sum(rr) / len(rr)


def dcg_at_k(retrieved: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Discounted cumulative gain using graded relevance ``gains`` (default 0)."""
    total = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        g = gains.get(item, 0.0)
        total += g / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Normalized DCG@k. ``gains`` maps event_id -> graded relevance (>=0)."""
    ideal_order = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_order, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gains, k) / idcg


def mean_recall_at_k(
    retrieved_lists: Sequence[Sequence[str]], relevant_sets: Sequence[set[str]], k: int
) -> float:
    if not retrieved_lists:
        return 0.0
    vals = [recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)]
    return sum(vals) / len(vals)


__all__ = [
    "recall_at_k",
    "reciprocal_rank",
    "mrr",
    "dcg_at_k",
    "ndcg_at_k",
    "mean_recall_at_k",
]
