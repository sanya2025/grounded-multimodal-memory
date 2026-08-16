"""Semantic retrieval over the memory store (vector similarity)."""

from __future__ import annotations

import numpy as np

from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore


class SemanticRetriever:
    """Rank events by embedding cosine similarity to the query embedding."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[MemoryEvent, float]]:
        return self.store.semantic_search(query_embedding, top_k=top_k)


__all__ = ["SemanticRetriever"]
