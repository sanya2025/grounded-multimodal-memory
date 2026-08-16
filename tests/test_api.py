"""API tests using the MockVLM system (no model downloads).

Skipped cleanly if FastAPI/starlette are not installed (base install).
"""

from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from grounded_memory.api.main import create_app  # noqa: E402
from grounded_memory.pipelines.inference import GroundedMemorySystem  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    app = create_app(GroundedMemorySystem())
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_events"] == 0


def test_observe_then_query_flow(client):
    # Observe three events.
    for i in range(3):
        r = client.post(
            "/observe",
            json={"timestamp": datetime(2026, 8, 14, 8 + i).isoformat(),
                  "image_path": f"frame_{i}.jpg"},
        )
        assert r.status_code == 200
        assert r.json()["event_id"].startswith("event_")

    health = client.get("/health").json()
    assert health["n_events"] == 3

    q = client.post("/query", json={"question": "What was observed?"})
    assert q.status_code == 200
    body = q.json()
    assert "answer" in body
    assert "evidence" in body
    assert isinstance(body["abstained"], bool)


def test_query_with_empty_memory_abstains(client):
    q = client.post("/query", json={"question": "Where are the keys?"})
    assert q.status_code == 200
    assert q.json()["abstained"] is True
