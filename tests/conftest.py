"""Shared fixtures. Tests must run without large datasets or model downloads."""

from __future__ import annotations

from datetime import datetime

import pytest

from grounded_memory.memory.embeddings import HashingTextEmbedder
from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore


@pytest.fixture
def embedder() -> HashingTextEmbedder:
    return HashingTextEmbedder(dim=128)


@pytest.fixture
def toy_scene_graph() -> dict:
    """A tiny GQA-style scene graph used across scene/selection/grounding tests."""
    return {
        "objects": {
            "o1": {
                "name": "umbrella",
                "attributes": ["black"],
                "relations": [{"name": "to the left of", "object": "o2"}],
            },
            "o2": {"name": "bicycle", "attributes": ["red"], "relations": []},
            "p1": {
                "name": "person",
                "attributes": [],
                "relations": [{"name": "holding", "object": "o1"}],
            },
        }
    }


@pytest.fixture
def toy_store(embedder: HashingTextEmbedder) -> MemoryStore:
    """A 3-event memory sequence (keys move mug -> backpack -> car)."""
    store = MemoryStore()
    events = [
        ("event_0001", "08:30", "A red coffee mug is on the kitchen counter.", ["mug", "counter"]),
        ("event_0002", "10:30", "Keys are placed inside a black backpack.", ["keys", "backpack"]),
        ("event_0003", "11:45", "The black backpack is placed in the car.", ["backpack", "car"]),
    ]
    for i, (eid, hhmm, desc, ents) in enumerate(events):
        h, m = map(int, hhmm.split(":"))
        store.add(
            MemoryEvent(
                event_id=eid,
                timestamp=datetime(2026, 8, 14, h, m),
                image_path=f"frame_{i}.jpg",
                description=desc,
                entities=ents,
                embedding=embedder.embed_text(desc),
                sequence_id="seq_0",
            )
        )
    return store
