"""Memory event schema for episodic multimodal memory.

Each observation becomes an event with a timestamp, a textual scene description,
extracted entities/relationships, and an embedding. Events are serializable so
memory can be persisted and reloaded reproducibly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class MemoryEvent:
    """A single stored observation."""

    event_id: str
    timestamp: datetime
    image_path: str
    description: str
    entities: list[str] = field(default_factory=list)
    relationships: list[dict[str, str]] = field(default_factory=list)
    embedding: np.ndarray | None = None
    sequence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def entity_set(self) -> set[str]:
        return {e.lower() for e in self.entities}

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        """JSON-serializable dict. Embeddings are stored separately by default."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.astype(float).tolist()
        else:
            d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEvent:
        ts = d["timestamp"]
        timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        emb = d.get("embedding")
        embedding = np.asarray(emb, dtype=np.float32) if emb is not None else None
        return cls(
            event_id=d["event_id"],
            timestamp=timestamp,
            image_path=d.get("image_path", ""),
            description=d.get("description", ""),
            entities=list(d.get("entities", [])),
            relationships=list(d.get("relationships", [])),
            embedding=embedding,
            sequence_id=d.get("sequence_id"),
            metadata=dict(d.get("metadata", {})),
        )


__all__ = ["MemoryEvent"]
