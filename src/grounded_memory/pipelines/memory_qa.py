"""Experiment-2 harness: compare memory/retrieval methods on temporal QA.

Given a temporal sequence dataset and a system, runs the four E2.1 methods
(no_memory, full_history, semantic, hybrid) and returns per-question retrieval
rankings + answers for downstream retrieval/QA metrics. Retrieval quality and
answer quality are recorded separately on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grounded_memory.memory.store import MemoryStore
from grounded_memory.retrieval.hybrid import HybridRetriever
from grounded_memory.retrieval.semantic import SemanticRetriever


@dataclass
class MethodOutput:
    method: str
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: set[str] = field(default_factory=set)
    context_event_ids: list[str] = field(default_factory=list)


def retrieve_for_question(
    method: str,
    store: MemoryStore,
    query_embedding: Any,
    query_entities: set[str],
    reference: Any = None,
    top_k: int = 5,
) -> MethodOutput:
    """Return the retrieved event ids for one method (E2.1)."""
    if method == "no_memory":
        return MethodOutput(method, retrieved_ids=[], context_event_ids=[])
    if method == "full_history":
        ids = [e.event_id for e in store.events]
        return MethodOutput(method, retrieved_ids=ids, context_event_ids=ids)
    if method == "semantic":
        pairs = SemanticRetriever(store).retrieve(query_embedding, top_k=top_k)
        ids = [e.event_id for e, _ in pairs]
        return MethodOutput(method, retrieved_ids=ids, context_event_ids=ids)
    if method == "hybrid":
        results = HybridRetriever(store).retrieve(
            query_embedding=query_embedding,
            query_entities=query_entities,
            reference=reference,
            top_k=top_k,
        )
        ids = [r.event.event_id for r in results]
        return MethodOutput(method, retrieved_ids=ids, context_event_ids=ids)
    raise ValueError(f"Unknown method {method!r}")


__all__ = ["MethodOutput", "retrieve_for_question"]
