"""Temporal and entity-aware retrieval components."""

from __future__ import annotations

from datetime import datetime

from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore
from grounded_memory.memory.temporal import temporal_relevance


class TemporalRetriever:
    """Rank events purely by temporal relevance to a reference time."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        reference: datetime | None,
        constraint: str = "recency",
        top_k: int = 5,
        half_life_s: float = 3600.0,
    ) -> list[tuple[MemoryEvent, float]]:
        scored = [
            (e, temporal_relevance(e, reference, constraint, half_life_s))
            for e in self.store.events
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]


def entity_overlap(query_entities: set[str], event: MemoryEvent) -> float:
    """Jaccard-style overlap between query entities and an event's entities."""
    q = {e.lower() for e in query_entities}
    if not q:
        return 0.0
    e = event.entity_set()
    if not e:
        return 0.0
    return len(q & e) / len(q | e)


__all__ = ["TemporalRetriever", "entity_overlap"]
