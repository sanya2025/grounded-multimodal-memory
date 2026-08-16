"""The end-to-end grounded multimodal memory system (observe + query).

Ties together scene extraction, embedding, the memory store, retrieval, and
grounded answering with provenance. Backs the FastAPI service. Works with any
``VisionLanguageModel`` (including ``MockVLM``) and any ``Embedder``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from grounded_memory.memory.embeddings import Embedder, HashingTextEmbedder
from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore
from grounded_memory.models.base import VisionLanguageModel
from grounded_memory.models.mock import MockVLM
from grounded_memory.prompts import build_memory_qa_prompt
from grounded_memory.retrieval.hybrid import HybridRetriever
from grounded_memory.scene.extractor import extract_scene


@dataclass
class ObserveResult:
    event_id: str
    description: str
    entities: list[str]
    latency_s: float


@dataclass
class QueryResult:
    answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    confidence: float | None = None  # model-reported, NOT calibrated
    component_latencies: dict[str, float] = field(default_factory=dict)


class GroundedMemorySystem:
    """Observe visual events into memory, then answer grounded temporal queries."""

    def __init__(
        self,
        model: VisionLanguageModel | None = None,
        embedder: Embedder | None = None,
        retriever_weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
        top_k: int = 5,
    ) -> None:
        self.model = model or MockVLM()
        self.embedder = embedder or HashingTextEmbedder()
        self.store = MemoryStore()
        a, b, g = retriever_weights
        self.retriever = HybridRetriever(self.store, alpha=a, beta=b, gamma=g)
        self.top_k = top_k
        self._counter = 0

    # ---- observe ----
    def observe(
        self, image: Any, timestamp: datetime | None = None, image_path: str = ""
    ) -> ObserveResult:
        t0 = time.perf_counter()
        ext = extract_scene(self.model, image, condition="structured")
        if ext.scene is not None:
            description = ext.scene.scene_summary or ext.raw_output
            entities = ext.scene.object_labels()
            relationships = [r.model_dump() for r in ext.scene.relationships]
        else:
            # Parser failed: still store the raw description; nothing discarded.
            description = ext.raw_output
            entities = []
            relationships = []
        embedding = self.embedder.embed_text(description)
        self._counter += 1
        event = MemoryEvent(
            event_id=f"event_{self._counter:04d}",
            timestamp=timestamp or _default_ts(self._counter),
            image_path=image_path,
            description=description,
            entities=entities,
            relationships=relationships,
            embedding=embedding,
        )
        self.store.add(event)
        return ObserveResult(
            event_id=event.event_id,
            description=description,
            entities=entities,
            latency_s=time.perf_counter() - t0,
        )

    # ---- query ----
    def query(
        self,
        question: str,
        reference: datetime | None = None,
        temporal_constraint: str = "recency",
        query_entities: set[str] | None = None,
    ) -> QueryResult:
        comp: dict[str, float] = {}
        t_embed = time.perf_counter()
        q_emb = self.embedder.embed_text(question)
        comp["memory_embedding"] = time.perf_counter() - t_embed

        t_ret = time.perf_counter()
        results = self.retriever.retrieve(
            query_embedding=q_emb,
            query_entities=query_entities or set(),
            reference=reference,
            temporal_constraint=temporal_constraint,
            top_k=self.top_k,
        )
        comp["retrieval"] = time.perf_counter() - t_ret

        if not results:
            return QueryResult(
                answer="The available visual observations are insufficient.",
                abstained=True,
                component_latencies=comp,
            )

        context = "\n".join(
            f"[{r.event.event_id} @ {r.event.timestamp.isoformat()}] {r.event.description}"
            for r in results
        )
        prompt = build_memory_qa_prompt(question, context)
        t_gen = time.perf_counter()
        gen = self.model.generate(None, prompt) if _accepts_text_only(self.model) else \
            self.model.generate(results[0].event.image_path or None, prompt)
        comp["generation"] = time.perf_counter() - t_gen

        answer = gen.text
        abstained = "insufficient" in answer.lower()
        evidence = [
            {"event_id": r.event.event_id, "timestamp": r.event.timestamp.isoformat(),
             "score": round(r.score, 4)}
            for r in results
        ]
        return QueryResult(
            answer=answer,
            evidence=[] if abstained else evidence,
            abstained=abstained,
            component_latencies=comp,
        )


def _default_ts(i: int) -> datetime:
    # Deterministic synthetic timestamps for offline/demo use (no wall clock).
    return datetime(2026, 1, 1, 8, 0, 0).replace(minute=(i * 5) % 60, hour=8 + (i * 5) // 60)


def _accepts_text_only(model: VisionLanguageModel) -> bool:
    return isinstance(model, MockVLM)


__all__ = ["GroundedMemorySystem", "ObserveResult", "QueryResult"]
