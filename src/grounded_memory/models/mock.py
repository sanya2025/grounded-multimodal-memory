"""Deterministic mock VLM for tests and offline pipeline wiring.

Returns canned but schema-valid output so the full scene/memory/QA pipelines and
the API can be exercised without downloading any model weights.
"""

from __future__ import annotations

import json
import time
from typing import Any

from grounded_memory.models.base import GenerationOutput, VisionLanguageModel


class MockVLM(VisionLanguageModel):
    """A tiny, dependency-free stand-in for a real VLM."""

    name = "mock_vlm"
    model_id = "mock/deterministic-vlm"
    revision = "v1"

    def __init__(self, canned_structured: dict[str, Any] | None = None) -> None:
        self._canned = canned_structured or {
            "scene_summary": "A person is holding a black umbrella next to a bicycle.",
            "entities": [
                {"id": "person_1", "label": "person", "attributes": [], "confidence": 0.9},
                {"id": "object_1", "label": "umbrella", "attributes": ["black"],
                 "confidence": 0.88},
                {"id": "object_2", "label": "bicycle", "attributes": [], "confidence": 0.8},
            ],
            "relationships": [
                {"subject": "person_1", "relation": "holding", "object": "object_1",
                 "confidence": 0.85}
            ],
            "spatial_relations": [
                {"subject": "person_1", "relation": "left_of", "object": "object_2",
                 "confidence": 0.7}
            ],
            "claims": [
                {"claim": "The person is holding a black umbrella.",
                 "evidence": "The umbrella is gripped in the person's hand.",
                 "evidence_type": "direct_or_supported", "confidence": 0.85}
            ],
            "uncertainty": [
                {"about": "brand of the umbrella", "reason": "No readable logo is visible."}
            ],
        }

    def generate(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        t0 = time.perf_counter()
        lowered = prompt.lower()
        if "json object" in lowered or "schema" in lowered:
            text = json.dumps(self._canned)
        elif "insufficient" in lowered or "evidence" in lowered:
            text = (
                "Visible entities: a person, a black umbrella, a bicycle. "
                "The person is holding the umbrella (direct evidence: hand grip). "
                "The brand cannot be determined; the available visual evidence is "
                "insufficient."
            )
        else:
            text = "A person stands holding a black umbrella beside a bicycle."
        latency = time.perf_counter() - t0
        return GenerationOutput(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_s=latency,
            component_latencies={"generation": latency},
            metadata={"mock": True},
        )


__all__ = ["MockVLM"]
