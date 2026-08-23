"""Tests for joining a structured SceneRepresentation to a GQA SceneGraph."""

from __future__ import annotations

from grounded_memory.data.gqa import SceneGraph
from grounded_memory.evaluation.grounding import EvidenceLabel
from grounded_memory.evaluation.structured_scene import (
    predicted_attribute_pairs,
    predicted_relationship_triples,
    predicted_spatial_triples,
    reference_attribute_pairs,
    reference_positional_triples,
    reference_relation_triples,
    score_structured_scene,
)
from grounded_memory.scene.schema import Entity, Relationship, SceneRepresentation, SpatialRelation


def _scene_graph() -> SceneGraph:
    """person: holding + above (non-positional) + to-the-left-of (positional), all -> umbrella."""
    return SceneGraph(
        "img1",
        {
            "objects": {
                "o1": {
                    "name": "person",
                    "attributes": ["tall"],
                    "relations": [
                        {"name": "holding", "object": "o2"},
                        {"name": "above", "object": "o2"},
                        {"name": "to the left of", "object": "o2"},
                    ],
                },
                "o2": {"name": "umbrella", "attributes": ["black"], "relations": []},
            }
        },
    )


def _matching_scene() -> SceneRepresentation:
    return SceneRepresentation(
        entities=[
            Entity(id="person_1", label="person", attributes=["tall"]),
            Entity(id="obj_1", label="umbrella", attributes=["black"]),
        ],
        relationships=[Relationship(subject="person_1", relation="holding", object="obj_1")],
        spatial_relations=[SpatialRelation(subject="person_1", relation="above", object="obj_1")],
    )


def test_predicted_attribute_pairs_from_entities():
    scene = _matching_scene()
    assert predicted_attribute_pairs(scene) == {("person", "tall"), ("umbrella", "black")}


def test_predicted_relationship_triples_resolves_entity_ids_to_labels():
    scene = _matching_scene()
    assert predicted_relationship_triples(scene) == {("person", "holding", "umbrella")}


def test_predicted_spatial_triples_resolves_entity_ids_to_labels():
    scene = _matching_scene()
    assert predicted_spatial_triples(scene) == {("person", "above", "umbrella")}


def test_predicted_relationship_triples_accepts_literal_labels_not_just_ids():
    scene = SceneRepresentation(
        entities=[Entity(id="person_1", label="person")],
        relationships=[Relationship(subject="person_1", relation="holding", object="a kite")],
    )
    assert predicted_relationship_triples(scene) == {("person", "holding", "a kite")}


def test_reference_attribute_pairs_from_scene_graph():
    sg = _scene_graph()
    assert reference_attribute_pairs(sg) == {("person", "tall"), ("umbrella", "black")}


def test_reference_relation_triples_excludes_positional_phrases():
    sg = _scene_graph()
    # "to the left of" is excluded -- see POSITIONAL_RELATION_PHRASES.
    assert reference_relation_triples(sg) == {
        ("person", "holding", "umbrella"),
        ("person", "above", "umbrella"),
    }


def test_reference_positional_triples_only_includes_left_right():
    sg = _scene_graph()
    assert reference_positional_triples(sg) == {("person", "to the left of", "umbrella")}


def test_score_structured_scene_relationships_and_spatial_share_non_positional_reference():
    result = score_structured_scene(_matching_scene(), _scene_graph())
    assert result.objects.precision == 1.0
    assert result.objects.recall == 1.0
    assert result.attributes.precision == 1.0
    assert result.attributes.recall == 1.0
    # relationship_accuracy and spatial_accuracy both score against the SAME
    # 2-item non-positional reference (holding + above). The fixture predicts
    # one triple in each category, so each finds 1 of 2 reference items:
    # perfect precision (nothing wrong was claimed), 0.5 recall (half found).
    assert result.relationships.precision == 1.0
    assert result.relationships.recall == 0.5
    assert result.spatial.precision == 1.0
    assert result.spatial.recall == 0.5
    # Neither predicted triple is positional, so positional recall is 0 even
    # though the model made no incorrect claims elsewhere.
    assert result.positional.recall == 0.0


def test_score_structured_scene_positional_scored_when_model_states_it():
    scene = SceneRepresentation(
        entities=[
            Entity(id="p1", label="person"),
            Entity(id="u1", label="umbrella"),
        ],
        spatial_relations=[SpatialRelation(subject="p1", relation="to the left of", object="u1")],
    )
    result = score_structured_scene(scene, _scene_graph())
    assert result.positional.precision == 1.0
    assert result.positional.recall == 1.0
    # The positional claim doesn't leak into the non-positional spatial score.
    assert result.spatial.recall == 0.0


def test_score_structured_scene_unannotated_object_is_not_verifiable_not_hallucination():
    scene = SceneRepresentation(entities=[Entity(id="d1", label="dog")])
    result = score_structured_scene(scene, _scene_graph())
    assert result.evidence_labels == [EvidenceLabel.NOT_VERIFIABLE]
    # Strict hallucination_rate only counts CONTRADICTED; an unannotated claim
    # must not be silently treated as a hallucination (project's core rule).
    assert result.hallucination_rate == 0.0
    assert result.objects.precision == 0.0


def test_score_structured_scene_missing_objects_hurt_recall():
    scene = SceneRepresentation(entities=[Entity(id="p1", label="person")])
    result = score_structured_scene(scene, _scene_graph())
    assert result.objects.recall == 0.5  # found person, missed umbrella
    assert result.objects.precision == 1.0


def test_score_structured_scene_wrong_color_shows_up_as_real_hallucination():
    # umbrella is annotated "black" in _scene_graph(); claiming "red" is a
    # genuine, detectable conflict (both are in the "color" category).
    scene = SceneRepresentation(
        entities=[Entity(id="u1", label="umbrella", attributes=["red"])],
    )
    result = score_structured_scene(scene, _scene_graph())
    assert EvidenceLabel.CONTRADICTED in result.evidence_labels
    # labels: object claim ("umbrella" is annotated) -> SUPPORTED,
    # attribute claim ("red" vs annotated "black") -> CONTRADICTED.
    # strict rate = contradicted / (supported + contradicted) = 1/2.
    assert result.hallucination_rate == 0.5


def test_score_structured_scene_wrong_spatial_direction_is_contradicted():
    # _scene_graph() annotates person "above" umbrella; claiming "below" is
    # the spatial opposite -- a real conflict, not just unverified.
    scene = SceneRepresentation(
        entities=[Entity(id="p1", label="person"), Entity(id="u1", label="umbrella")],
        spatial_relations=[SpatialRelation(subject="p1", relation="below", object="u1")],
    )
    result = score_structured_scene(scene, _scene_graph())
    assert EvidenceLabel.CONTRADICTED in result.evidence_labels


def test_score_structured_scene_empty_prediction_gives_zeroed_not_crashed():
    result = score_structured_scene(SceneRepresentation(), _scene_graph())
    assert result.objects.f1 == 0.0
    assert result.evidence_labels == []
    assert result.hallucination_rate == 0.0
    assert result.evidence_support_rate == 0.0
