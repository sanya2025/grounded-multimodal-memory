"""Configurable hybrid retrieval: semantic + temporal + entity signals.

    score = alpha * semantic_similarity
          + beta  * temporal_relevance
          + gamma * entity_overlap

Weights are configurable and are a *starting point*, not claimed optimal. Tune on
a development subset or treat the sweep as an experiment (E2.4) — never tune on
the frozen evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from grounded_memory.memory.embeddings import cosine_similarity
from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore
from grounded_memory.memory.temporal import temporal_relevance
from grounded_memory.retrieval.temporal import entity_overlap


@dataclass
class RetrievalResult:
    event: MemoryEvent
    score: float
    components: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ) -> None:
        self.store = store
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieve(
        self,
        query_embedding: np.ndarray | None = None,
        query_entities: set[str] | None = None,
        reference: datetime | None = None,
        temporal_constraint: str = "recency",
        top_k: int = 5,
        half_life_s: float = 3600.0,
    ) -> list[RetrievalResult]:
        """Rank events by the weighted combination of enabled signals.

        Any signal whose inputs are absent contributes 0, so the same method
        supports the E2.4 ablation (semantic only, +time, +entities, +both) just
        by varying which arguments/weights are provided.
        """
        query_entities = query_entities or set()
        results: list[RetrievalResult] = []
        for event in self.store.events:
            sem = (
                cosine_similarity(query_embedding, event.embedding)
                if query_embedding is not None and event.embedding is not None
                else 0.0
            )
            temp = (
                temporal_relevance(event, reference, temporal_constraint, half_life_s)
                if reference is not None
                else 0.0
            )
            ent = entity_overlap(query_entities, event) if query_entities else 0.0
            score = self.alpha * sem + self.beta * temp + self.gamma * ent
            results.append(
                RetrievalResult(
                    event=event,
                    score=score,
                    components={"semantic": sem, "temporal": temp, "entity": ent},
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


__all__ = ["HybridRetriever", "RetrievalResult"]
