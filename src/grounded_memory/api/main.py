"""FastAPI app for the grounded multimodal memory system.

Endpoints:
- GET  /health   -> liveness
- POST /observe  -> run scene extraction, store a memory event, return event id
- POST /query    -> retrieve evidence, produce a grounded answer with provenance

The app is constructed via ``create_app`` so tests can inject a ``MockVLM`` system
(no model downloads). Storage is simple and local for the first version.

Run:  uvicorn grounded_memory.api.main:app --reload   (requires the 'api' extra)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from grounded_memory.pipelines.inference import GroundedMemorySystem


def create_app(system: GroundedMemorySystem | None = None):
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="Grounded Multimodal Scene Memory",
        version="0.1.0",
        description="Observe visual events into memory and answer grounded temporal queries.",
    )
    app.state.system = system or GroundedMemorySystem()

    class ObserveRequest(BaseModel):
        timestamp: datetime | None = None
        image_path: str = Field(default="", description="Local/uploaded image reference.")

    class ObserveResponse(BaseModel):
        event_id: str
        description: str
        entities: list[str]
        latency_s: float

    class QueryRequest(BaseModel):
        question: str
        reference: datetime | None = None
        temporal_constraint: str = "recency"
        query_entities: list[str] = Field(default_factory=list)

    class Evidence(BaseModel):
        event_id: str
        timestamp: str
        score: float

    class QueryResponse(BaseModel):
        answer: str
        evidence: list[Evidence] = Field(default_factory=list)
        abstained: bool = False
        confidence: float | None = None  # model-reported, NOT calibrated

    @app.get("/health")
    def health() -> dict[str, Any]:
        sys_: GroundedMemorySystem = app.state.system
        return {"status": "ok", "n_events": len(sys_.store), "model": sys_.model.name}

    @app.post("/observe", response_model=ObserveResponse)
    def observe(req: ObserveRequest) -> ObserveResponse:
        sys_: GroundedMemorySystem = app.state.system
        image = req.image_path or None
        result = sys_.observe(image, timestamp=req.timestamp, image_path=req.image_path)
        return ObserveResponse(**result.__dict__)

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        sys_: GroundedMemorySystem = app.state.system
        result = sys_.query(
            req.question,
            reference=req.reference,
            temporal_constraint=req.temporal_constraint,
            query_entities=set(req.query_entities),
        )
        return QueryResponse(
            answer=result.answer,
            evidence=[Evidence(**e) for e in result.evidence],
            abstained=result.abstained,
            confidence=result.confidence,
        )

    return app


# Module-level app for `uvicorn grounded_memory.api.main:app`.
# Guarded so importing this module never fails when FastAPI is absent.
try:  # pragma: no cover - exercised via create_app in tests
    app = create_app()
except Exception:  # noqa: BLE001
    app = None


__all__ = ["create_app", "app"]
