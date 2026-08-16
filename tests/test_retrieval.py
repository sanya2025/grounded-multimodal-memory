"""Tests for memory store, retrieval strategies, and retrieval metrics."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from grounded_memory.evaluation.retrieval import (
    mrr,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from grounded_memory.memory.temporal import temporal_relevance
from grounded_memory.retrieval.hybrid import HybridRetriever
from grounded_memory.retrieval.semantic import SemanticRetriever
from grounded_memory.retrieval.temporal import entity_overlap


def test_semantic_retrieval_finds_relevant_event(toy_store, embedder):
    q = embedder.embed_text("Where is the backpack?")
    pairs = SemanticRetriever(toy_store).retrieve(q, top_k=2)
    ids = [e.event_id for e, _ in pairs]
    # backpack appears in events 2 and 3; at least one should rank top-2.
    assert {"event_0002", "event_0003"} & set(ids)


def test_store_persistence_roundtrip(tmp_path, toy_store):
    jsonl = tmp_path / "events.jsonl"
    npy = tmp_path / "emb.npy"
    toy_store.save(jsonl, npy)
    from grounded_memory.memory.store import MemoryStore

    loaded = MemoryStore.load(jsonl, npy)
    assert len(loaded) == len(toy_store)
    assert loaded.get("event_0002") is not None
    assert loaded.get("event_0002").embedding is not None


def test_entity_overlap():
    from grounded_memory.memory.events import MemoryEvent

    ev = MemoryEvent("e", datetime(2026, 1, 1), "", "", entities=["keys", "backpack"])
    assert entity_overlap({"keys"}, ev) > 0
    assert entity_overlap(set(), ev) == 0.0
    assert entity_overlap({"unicorn"}, ev) == 0.0


def test_temporal_relevance_before_after():
    from grounded_memory.memory.events import MemoryEvent

    ref = datetime(2026, 8, 14, 10, 0)
    early = MemoryEvent("a", datetime(2026, 8, 14, 9, 0), "", "")
    late = MemoryEvent("b", datetime(2026, 8, 14, 11, 0), "", "")
    assert temporal_relevance(early, ref, "before") > 0
    assert temporal_relevance(late, ref, "before") == 0.0
    assert temporal_relevance(late, ref, "after") == 1.0
    assert temporal_relevance(early, ref, "after") == 0.0


def test_hybrid_weighting_prefers_entity_and_time(toy_store, embedder):
    q = embedder.embed_text("backpack")
    ref = datetime(2026, 8, 14, 12, 0)
    results = HybridRetriever(toy_store, alpha=0.4, beta=0.3, gamma=0.3).retrieve(
        query_embedding=q,
        query_entities={"backpack"},
        reference=ref,
        temporal_constraint="recency",
        top_k=3,
    )
    assert results[0].event.event_id in {"event_0002", "event_0003"}
    # component breakdown is populated
    assert set(results[0].components) == {"semantic", "temporal", "entity"}


def test_recall_at_k():
    retrieved = ["e3", "e1", "e5", "e2"]
    relevant = {"e1", "e2"}
    assert recall_at_k(retrieved, relevant, k=2) == 0.5   # only e1 in top-2
    assert recall_at_k(retrieved, relevant, k=4) == 1.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert math.isclose(mrr([["a", "b"], ["c", "d"]], [{"b"}, {"c"}]), (0.5 + 1.0) / 2)


def test_ndcg_at_k_perfect_and_imperfect():
    gains = {"a": 3.0, "b": 2.0, "c": 1.0}
    # ideal order gives ndcg 1.0
    assert math.isclose(ndcg_at_k(["a", "b", "c"], gains, k=3), 1.0)
    # reversed order should be < 1
    assert ndcg_at_k(["c", "b", "a"], gains, k=3) < 1.0


def test_numpy_fallback_matches_when_no_faiss(toy_store, embedder):
    # Force the NumPy path by clearing any faiss index.
    toy_store._faiss_index = None
    toy_store._dirty = True
    q = embedder.embed_text("mug on the counter")
    pairs = toy_store.semantic_search(q, top_k=1)
    assert pairs and pairs[0][0].event_id == "event_0001"
    assert isinstance(pairs[0][1], float)
    assert np.isfinite(pairs[0][1])
