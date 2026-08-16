"""In-memory episodic store with optional FAISS acceleration.

Holds ``MemoryEvent`` objects, maintains an embedding matrix for semantic search,
and persists to / loads from JSONL + .npy. If ``faiss`` is installed the semantic
search uses a flat index; otherwise it falls back to exact NumPy cosine search.
Both paths return identical rankings for normalized vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grounded_memory.memory.embeddings import l2_normalize
from grounded_memory.memory.events import MemoryEvent


class MemoryStore:
    def __init__(self, dim: int | None = None) -> None:
        self._events: list[MemoryEvent] = []
        self._dim = dim
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._dirty = True

    # ---- population ----
    def add(self, event: MemoryEvent) -> None:
        if event.embedding is not None:
            d = int(event.embedding.shape[0])
            if self._dim is None:
                self._dim = d
            elif d != self._dim:
                raise ValueError(f"Embedding dim {d} != store dim {self._dim}")
        self._events.append(event)
        self._dirty = True

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[MemoryEvent]:
        return list(self._events)

    def get(self, event_id: str) -> MemoryEvent | None:
        return next((e for e in self._events if e.event_id == event_id), None)

    # ---- index ----
    def _rebuild(self) -> None:
        embedded = [e for e in self._events if e.embedding is not None]
        if not embedded:
            self._matrix = None
            self._faiss_index = None
            self._dirty = False
            return
        mat = np.vstack([l2_normalize(e.embedding) for e in embedded]).astype(np.float32)
        self._matrix = mat
        self._embedded_events = embedded
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(mat.shape[1])  # inner product == cosine for L2-normed
            index.add(mat)
            self._faiss_index = index
        except Exception:  # noqa: BLE001 - FAISS optional; NumPy fallback is fine
            self._faiss_index = None
        self._dirty = False

    def semantic_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[MemoryEvent, float]]:
        """Return up to ``top_k`` (event, cosine_similarity) pairs, best first."""
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return []
        q = l2_normalize(np.asarray(query_embedding, dtype=np.float32)).reshape(1, -1)
        k = min(top_k, self._matrix.shape[0])
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, k)
            pairs = [(self._embedded_events[i], float(s)) for i, s in zip(idxs[0], scores[0])]
        else:
            sims = (self._matrix @ q.T).ravel()
            order = np.argsort(-sims)[:k]
            pairs = [(self._embedded_events[i], float(sims[i])) for i in order]
        return pairs

    # ---- persistence ----
    def save(self, jsonl_path: str | Path, embeddings_path: str | Path | None = None) -> None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for e in self._events:
                fh.write(json.dumps(e.to_dict(include_embedding=False)) + "\n")
        if embeddings_path is not None:
            embs = [e.embedding for e in self._events if e.embedding is not None]
            if embs:
                np.save(Path(embeddings_path), np.vstack(embs).astype(np.float32))

    @classmethod
    def load(
        cls, jsonl_path: str | Path, embeddings_path: str | Path | None = None
    ) -> MemoryStore:
        store = cls()
        rows = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line]
        matrix = None
        if embeddings_path is not None and Path(embeddings_path).exists():
            matrix = np.load(embeddings_path)
        for i, row in enumerate(rows):
            event = MemoryEvent.from_dict(row)
            if matrix is not None and i < len(matrix):
                event.embedding = matrix[i]
            store.add(event)
        return store


__all__ = ["MemoryStore"]
