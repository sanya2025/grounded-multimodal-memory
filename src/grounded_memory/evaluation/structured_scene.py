"""Join one structured-condition SceneRepresentation to its GQA scene graph and score it.

This is the orchestration layer the other evaluation/ modules were missing: they
score plain predicted/reference sets, but nothing turned a parsed model
prediction and a GQA scene graph into those sets. Only the structured (JSON)
condition can be scored this way -- standard/grounded predictions are free text
with no extraction path (see notes/done_so_far.md for why that's out of scope
here).

Relationship subject/object fields in SceneRepresentation are the MODEL's own
entity ids (e.g. "person_1"), not labels -- they must be resolved via the
entities list before comparing against GQA object names.

GQA scene graphs don't separate spatial from non-spatial relations (a single
flat list per object), so relationship_accuracy and spatial_accuracy are each
scored against the SAME GQA relation-triple reference set. This module does
not attempt to classify GQA's own relations into spatial vs. non-spatial.

IMPORTANT: "to the left of" / "to the right of" are excluded from that
reference set by default. Measured across the frozen 120-image manifest, GQA
annotates these two phrases exhaustively between nearly every object pair --
they account for 35,150 of 36,714 total relation instances (95.7%), versus
~1,564 for everything else (wearing, holding, sitting_on, ...) combined. A
concise, reasonable model answer could never approach recall against a
reference set that dense, so including them there made relationship/spatial
accuracy measure "did you enumerate GQA's exhaustive pairwise grid" rather
than "did you get the scene's relations right". They are NOT discarded --
they're scored separately as `positional` (see StructuredSceneScore).
"""

from __future__ import annotations

from dataclasses import dataclass

from grounded_memory.data.gqa import SceneGraph
from grounded_memory.evaluation.attributes import attribute_accuracy
from grounded_memory.evaluation.grounding import (
    EvidenceLabel,
    verify_object_claim,
    verify_relation_claim,
)
from grounded_memory.evaluation.hallucination import evidence_support_rate, hallucination_rate
from grounded_memory.evaluation.objects import PRF1, object_prf1
from grounded_memory.evaluation.relationships import relationship_accuracy
from grounded_memory.evaluation.spatial import spatial_accuracy
from grounded_memory.scene.schema import SceneRepresentation

#: GQA's raw phrases for exhaustive pairwise left/right position -- excluded
#: from the main relation reference set; see module docstring.
POSITIONAL_RELATION_PHRASES = frozenset({"to the left of", "to the right of"})


@dataclass
class StructuredSceneScore:
    """Everything computed for one (SceneRepresentation, SceneGraph) pair."""

    objects: PRF1
    attributes: PRF1
    relationships: PRF1
    spatial: PRF1
    positional: PRF1
    evidence_labels: list[EvidenceLabel]
    hallucination_rate: float
    evidence_support_rate: float


def _entity_labels_by_id(scene: SceneRepresentation) -> dict[str, str]:
    return {e.id: e.label for e in scene.entities}


def _resolve(entity_id_or_label: str, labels_by_id: dict[str, str]) -> str:
    """Relationship subject/object may be an entity id OR already a literal label."""
    return labels_by_id.get(entity_id_or_label, entity_id_or_label)


def predicted_attribute_pairs(scene: SceneRepresentation) -> set[tuple[str, str]]:
    return {(e.label, attr) for e in scene.entities for attr in e.attributes}


def predicted_relationship_triples(scene: SceneRepresentation) -> set[tuple[str, str, str]]:
    labels = _entity_labels_by_id(scene)
    return {
        (_resolve(r.subject, labels), r.relation, _resolve(r.object, labels))
        for r in scene.relationships
    }


def predicted_spatial_triples(scene: SceneRepresentation) -> set[tuple[str, str, str]]:
    labels = _entity_labels_by_id(scene)
    return {
        (_resolve(r.subject, labels), r.relation, _resolve(r.object, labels))
        for r in scene.spatial_relations
    }


def reference_attribute_pairs(scene_graph: SceneGraph) -> set[tuple[str, str]]:
    return {
        (obj.get("name", ""), attr)
        for obj in scene_graph.objects.values()
        for attr in (obj.get("attributes") or [])
    }


def _all_reference_relation_triples(scene_graph: SceneGraph) -> set[tuple[str, str, str]]:
    objects = scene_graph.objects
    triples: set[tuple[str, str, str]] = set()
    for obj in objects.values():
        subj_name = obj.get("name", "")
        for rel in obj.get("relations") or []:
            target = objects.get(rel.get("object", ""), {})
            triples.add((subj_name, rel.get("name", ""), target.get("name", "")))
    return triples


def reference_relation_triples(scene_graph: SceneGraph) -> set[tuple[str, str, str]]:
    """GQA relations EXCLUDING the exhaustive left/right positional pairs (see module docstring)."""
    return {
        t for t in _all_reference_relation_triples(scene_graph)
        if t[1].strip().lower() not in POSITIONAL_RELATION_PHRASES
    }


def reference_positional_triples(scene_graph: SceneGraph) -> set[tuple[str, str, str]]:
    """Only the excluded exhaustive left/right positional pairs."""
    return {
        t for t in _all_reference_relation_triples(scene_graph)
        if t[1].strip().lower() in POSITIONAL_RELATION_PHRASES
    }


def score_structured_scene(
    scene: SceneRepresentation, scene_graph: SceneGraph
) -> StructuredSceneScore:
    """Score one structured-condition prediction against its GQA ground truth."""
    reference_relations = reference_relation_triples(scene_graph)
    reference_positional = reference_positional_triples(scene_graph)
    reference_attrs = reference_attribute_pairs(scene_graph)
    predicted_relationships = predicted_relationship_triples(scene)
    predicted_spatial = predicted_spatial_triples(scene)
    labels_by_id = _entity_labels_by_id(scene)

    labels: list[EvidenceLabel] = [
        verify_object_claim(e.label, scene_graph.raw) for e in scene.entities
    ]
    for r in (*scene.relationships, *scene.spatial_relations):
        labels.append(
            verify_relation_claim(
                _resolve(r.subject, labels_by_id), r.relation, _resolve(r.object, labels_by_id),
                scene_graph.raw,
            )
        )

    return StructuredSceneScore(
        objects=object_prf1(scene.object_labels(), scene_graph.object_names()),
        attributes=attribute_accuracy(predicted_attribute_pairs(scene), reference_attrs),
        relationships=relationship_accuracy(predicted_relationships, reference_relations),
        spatial=spatial_accuracy(predicted_spatial, reference_relations),
        # A model may file a "left of"/"right of" claim under either category,
        # so both are checked against the positional reference set together.
        positional=relationship_accuracy(
            predicted_relationships | predicted_spatial, reference_positional
        ),
        evidence_labels=labels,
        hallucination_rate=hallucination_rate(labels),
        evidence_support_rate=evidence_support_rate(labels),
    )


__all__ = [
    "StructuredSceneScore",
    "score_structured_scene",
    "predicted_attribute_pairs",
    "predicted_relationship_triples",
    "predicted_spatial_triples",
    "reference_attribute_pairs",
    "reference_relation_triples",
    "reference_positional_triples",
]
