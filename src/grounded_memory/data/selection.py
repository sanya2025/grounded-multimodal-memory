"""Framework for building the frozen 120-image GQA benchmark manifest.

Six buckets of 20 images (A-F). Selection uses scene-graph metadata to score each
candidate's fit for a bucket; a human-review column MUST be filled before the
manifest is frozen (do not blindly trust automatic selection). The scorer is
pure/deterministic so it is unit-testable without downloading GQA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grounded_memory.data.gqa import SceneGraph

# Bucket definitions mirror configs/datasets.yaml.
BUCKETS: dict[str, str] = {
    "A_multi_object": "at least 5 meaningful objects",
    "B_attributes": "color, material, size, state",
    "C_spatial": "left/right/front/behind/above/below/near",
    "D_human_object": "holding, wearing, using, sitting on, carrying",
    "E_complex_relational": "multiple interacting relationships",
    "F_hard_ambiguous": "clutter, occlusion, small objects, plausible-but-unsupported",
}

_SPATIAL_TERMS = {
    "left", "right", "front", "behind", "above", "below", "near", "under", "on top of"
}
_INTERACTION_TERMS = {
    "holding", "wearing", "using", "sitting on", "carrying", "riding", "eating", "looking at"
}
_HUMAN_LABELS = {"person", "man", "woman", "boy", "girl", "child", "people"}


@dataclass
class ManifestRow:
    image_id: str
    split: str
    primary_bucket: str
    num_objects: int
    num_relations: int
    num_attributes: int
    selection_reason: str
    scene_graph_id: str = ""
    source: str = "gqa"
    license_ref: str = "GQA / Visual Genome terms"
    manually_reviewed: bool = False
    ambiguity_notes: str = ""
    expected_question_types: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def csv_header() -> list[str]:
        return [
            "image_id", "split", "primary_bucket", "num_objects", "num_relations",
            "num_attributes", "selection_reason", "scene_graph_id", "source",
            "license_ref", "manually_reviewed", "ambiguity_notes",
            "expected_question_types",
        ]

    def to_csv_row(self) -> list[str]:
        return [
            self.image_id, self.split, self.primary_bucket, str(self.num_objects),
            str(self.num_relations), str(self.num_attributes), self.selection_reason,
            self.scene_graph_id, self.source, self.license_ref,
            str(self.manually_reviewed), self.ambiguity_notes,
            self.expected_question_types,
        ]


def bucket_features(sg: SceneGraph) -> dict[str, float]:
    """Compute per-bucket fit scores (higher = better fit) for a scene graph."""
    n_obj = sg.num_objects()
    n_rel = sg.num_relations()
    n_attr = sg.num_attributes()

    spatial_hits = 0
    interaction_hits = 0
    has_human = False
    for obj in sg.objects.values():
        name = obj.get("name", "").lower()
        if name in _HUMAN_LABELS:
            has_human = True
        for rel in obj.get("relations", []):
            rn = rel.get("name", "").lower()
            if any(term in rn for term in _SPATIAL_TERMS):
                spatial_hits += 1
            if any(term in rn for term in _INTERACTION_TERMS):
                interaction_hits += 1

    return {
        "A_multi_object": float(n_obj),
        "B_attributes": float(n_attr),
        "C_spatial": float(spatial_hits),
        "D_human_object": float(interaction_hits) + (2.0 if has_human else 0.0),
        "E_complex_relational": float(n_rel),
        # Hard/ambiguous heuristic: many small objects & relations, high clutter.
        "F_hard_ambiguous": float(n_obj) * 0.3 + float(n_rel) * 0.3,
    }


def assign_bucket(sg: SceneGraph) -> tuple[str, str]:
    """Assign the single best-fit primary bucket, returning (bucket, reason)."""
    scores = bucket_features(sg)
    bucket = max(scores, key=lambda k: scores[k])
    reason = f"top bucket score {bucket}={scores[bucket]:.1f}; " + ", ".join(
        f"{k}={v:.1f}" for k, v in scores.items()
    )
    return bucket, reason


def build_candidate_manifest(
    scene_graphs: dict[str, SceneGraph],
    split: str,
    per_bucket: int = 20,
    min_objects_A: int = 5,
) -> list[ManifestRow]:
    """Build a *candidate* manifest (needs manual review before freezing).

    Greedy: score every image for every bucket, then fill each bucket with its
    highest-scoring not-yet-used images. One image -> exactly one primary bucket.
    """
    scored: dict[str, dict[str, float]] = {
        img_id: bucket_features(sg) for img_id, sg in scene_graphs.items()
    }
    used: set[str] = set()
    rows: list[ManifestRow] = []

    for bucket in BUCKETS:
        candidates = sorted(
            (img for img in scene_graphs if img not in used),
            key=lambda img: scored[img][bucket],
            reverse=True,
        )
        if bucket == "A_multi_object":
            candidates = [c for c in candidates if scene_graphs[c].num_objects() >= min_objects_A]
        for img_id in candidates[:per_bucket]:
            sg = scene_graphs[img_id]
            used.add(img_id)
            rows.append(
                ManifestRow(
                    image_id=img_id,
                    split=split,
                    primary_bucket=bucket,
                    num_objects=sg.num_objects(),
                    num_relations=sg.num_relations(),
                    num_attributes=sg.num_attributes(),
                    selection_reason=f"auto-selected for {bucket} "
                    f"(score={scored[img_id][bucket]:.1f})",
                    scene_graph_id=img_id,
                    manually_reviewed=False,
                    expected_question_types=BUCKETS[bucket],
                )
            )
    return rows


__all__ = ["BUCKETS", "ManifestRow", "bucket_features", "assign_bucket", "build_candidate_manifest"]
