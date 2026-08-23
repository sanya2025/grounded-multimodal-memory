"""Tests for the scene schema, structured-output parsing, and probe generation."""

from __future__ import annotations

import json

from grounded_memory.scene.claims import parse_scene_output
from grounded_memory.scene.grounding import ProbeType, generate_probes
from grounded_memory.scene.schema import EvidenceType, SceneRepresentation


def test_scene_representation_roundtrip():
    scene = SceneRepresentation.model_validate(
        {
            "scene_summary": "A woman uses a laptop.",
            "entities": [{"id": "person_1", "label": "woman", "confidence": 0.9}],
            "claims": [
                {
                    "claim": "The woman is using the laptop.",
                    "evidence": "hands over keyboard",
                    "evidence_type": "direct_or_supported",
                    "confidence": 0.88,
                }
            ],
        }
    )
    assert scene.object_labels() == ["woman"]
    assert len(scene.supported_claims()) == 1


def test_parse_valid_json_blob():
    payload = {"scene_summary": "x", "entities": [], "claims": []}
    raw = f"Sure, here is the JSON:\n```json\n{json.dumps(payload)}\n```\nDone."
    result = parse_scene_output(raw)
    assert result.parse_ok
    assert result.scene is not None
    assert result.scene.scene_summary == "x"


def test_parse_failure_preserves_raw_output():
    raw = "I cannot produce JSON right now."
    result = parse_scene_output(raw)
    assert not result.parse_ok
    assert result.scene is None
    assert result.raw_output == raw  # nothing discarded
    assert result.error


def test_parse_balanced_braces_without_fence():
    raw = 'prefix {"scene_summary": "hi", "entities": [], "claims": []} suffix'
    result = parse_scene_output(raw)
    assert result.parse_ok
    assert result.scene.scene_summary == "hi"


def test_scene_representation_tolerates_extra_keys_from_model_drift():
    # extra="ignore": an open-weights model inventing an unexpected field
    # (e.g. "raw_confidence") must not fail the whole parse -- it's silently
    # dropped, not rejected. The original raw text is preserved separately
    # regardless (PredictionRecord.prediction), so nothing is lost either way.
    scene = SceneRepresentation.model_validate(
        {
            "scene_summary": "A dog runs.",
            "entities": [
                {"id": "dog_1", "label": "dog", "raw_confidence": "very high"}
            ],
            "claims": [],
            "extra_top_level_field": "some model invented this",
        }
    )
    assert scene.object_labels() == ["dog"]
    assert not hasattr(scene.entities[0], "raw_confidence")


def test_evidence_type_enum_values():
    assert EvidenceType.DIRECT.value == "direct_or_supported"
    assert EvidenceType.INSUFFICIENT.value == "insufficient_evidence"


def test_generate_probes_from_scene_graph(toy_scene_graph):
    probes = generate_probes(toy_scene_graph)
    types = {p.probe_type for p in probes}
    assert ProbeType.OBJECT in types
    assert ProbeType.ATTRIBUTE in types
    assert any(p.is_counterfactual for p in probes)
    # Positive object probe for umbrella must expect "yes".
    umbrella = [p for p in probes if "umbrella" in p.question and p.probe_type == ProbeType.OBJECT]
    assert umbrella and umbrella[0].expected_answer == "yes"
    # Absent-object counterfactual expects "no".
    absent = [p for p in probes if p.is_counterfactual and p.probe_type == ProbeType.OBJECT]
    assert absent and absent[0].expected_answer == "no"


def test_probes_preserve_source_facts(toy_scene_graph):
    probes = generate_probes(toy_scene_graph)
    holding = [p for p in probes if p.source_facts.get("relation") == "holding"]
    assert holding  # relation probe retains the scene-graph fact
