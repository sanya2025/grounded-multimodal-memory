"""End-to-end smoke test of the mock pipeline (integration)."""

from __future__ import annotations

import pytest

from grounded_memory.models.mock import MockVLM
from grounded_memory.pipelines.inference import GroundedMemorySystem
from grounded_memory.pipelines.scene_understanding import run_conditions_for_image
from grounded_memory.scene.extractor import extract_scene


@pytest.mark.integration
def test_extract_structured_scene_with_mock():
    ext = extract_scene(MockVLM(), image=None, condition="structured")
    assert ext.parse_ok
    assert ext.scene is not None
    assert "umbrella" in ext.scene.object_labels()


@pytest.mark.integration
def test_run_three_conditions_produces_records():
    records = run_conditions_for_image(
        MockVLM(), image=None, sample_id="img1",
        experiment_id="E1", run_id="run_test",
    )
    assert len(records) == 3
    conditions = {r.metrics["condition"] for r in records}
    assert conditions == {"standard", "grounded", "structured"}
    # Every record retains raw prediction text.
    assert all(r.prediction for r in records)
    # GenerationOutput.metadata survives into the persisted record (backend
    # extras like Ollama's thinking_fallback must not be silently dropped).
    assert all(isinstance(r.generation_metadata, dict) for r in records)


@pytest.mark.integration
def test_observe_query_system():
    system = GroundedMemorySystem()
    for _ in range(3):
        system.observe(image=None)
    assert len(system.store) == 3
    result = system.query("What is in the scene?")
    assert result.answer
