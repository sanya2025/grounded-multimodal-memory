"""Tests for manifest selection helpers (no GQA download required)."""

from __future__ import annotations

from grounded_memory.data.gqa import SceneGraph
from grounded_memory.data.selection import (
    BUCKETS,
    assign_bucket,
    bucket_features,
    build_candidate_manifest,
)


def _sg(objects: dict) -> SceneGraph:
    return SceneGraph("img", {"objects": objects})


def test_bucket_features_counts():
    sg = _sg(
        {
            "p1": {"name": "person", "attributes": [],
                   "relations": [{"name": "holding", "object": "o1"}]},
            "o1": {"name": "umbrella", "attributes": ["black"],
                   "relations": [{"name": "to the left of", "object": "o2"}]},
            "o2": {"name": "bicycle", "attributes": [], "relations": []},
        }
    )
    feats = bucket_features(sg)
    assert set(feats) == set(BUCKETS)
    assert feats["D_human_object"] > 0     # has human + interaction
    assert feats["C_spatial"] > 0          # "to the left of"


def test_assign_bucket_returns_valid_bucket():
    sg = _sg({f"o{i}": {"name": f"obj{i}", "attributes": [], "relations": []} for i in range(6)})
    bucket, reason = assign_bucket(sg)
    assert bucket in BUCKETS
    assert reason


def test_build_candidate_manifest_one_bucket_per_image():
    # 40 synthetic images so each 20-slot bucket can be attempted.
    scene_graphs = {}
    for i in range(40):
        objs = {
            f"o{j}": {
                "name": f"obj{j}",
                "attributes": ["red"] if i % 2 else [],
                "relations": [{"name": "to the left of", "object": "o0"}] if j > 0 else [],
            }
            for j in range(2 + (i % 6))
        }
        # inject a person into some
        if i % 3 == 0:
            objs["p"] = {"name": "person", "attributes": [],
                         "relations": [{"name": "holding", "object": "o0"}]}
        scene_graphs[str(i)] = SceneGraph(str(i), {"objects": objs})

    rows = build_candidate_manifest(scene_graphs, split="val", per_bucket=5)
    assigned = [r.image_id for r in rows]
    assert len(assigned) == len(set(assigned))  # no image used twice
    assert all(r.primary_bucket in BUCKETS for r in rows)
    assert all(r.manually_reviewed is False for r in rows)  # review still required


def test_manifest_row_csv_header_matches_row_length():
    from grounded_memory.data.selection import ManifestRow

    row = ManifestRow("1", "val", "A_multi_object", 5, 3, 2, "reason")
    assert len(row.to_csv_row()) == len(ManifestRow.csv_header())
