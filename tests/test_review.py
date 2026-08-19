"""Tests for the manual-review helpers (no GQA download, no real image needed
for the logic tests -- see grounded_memory.data.review)."""

from __future__ import annotations

import pytest

from grounded_memory.data.gqa import SceneGraph
from grounded_memory.data.review import (
    _boxes_to_draw,
    append_rejection,
    load_manifest,
    load_rejected_ids,
    load_rejection_log,
    reject_and_replace,
    save_manifest,
    scene_graph_to_triples,
)
from grounded_memory.data.selection import ManifestRow


def _sg(objects: dict) -> SceneGraph:
    return SceneGraph("img", {"objects": objects})


def _spatial_sg(image_id: str, spatial_hits: int) -> SceneGraph:
    return SceneGraph(
        image_id,
        {
            "objects": {
                f"o{i}": {
                    "name": f"obj{i}", "attributes": [],
                    "relations": [{"name": "to the left of", "object": "o0"}],
                }
                for i in range(spatial_hits)
            }
        },
    )


def _manifest_row(image_id: str, bucket: str = "C_spatial") -> ManifestRow:
    return ManifestRow(
        image_id=image_id, split="val", primary_bucket=bucket,
        num_objects=1, num_relations=1, num_attributes=0,
        selection_reason="auto-selected", scene_graph_id=image_id,
        manually_reviewed=True, ambiguity_notes="",
    )


def test_scene_graph_to_triples_attributes_and_relations():
    sg = _sg(
        {
            "o1": {"name": "umbrella", "attributes": ["black"],
                   "relations": [], "x": 0, "y": 0, "w": 10, "h": 10},
            "p1": {"name": "person", "attributes": [],
                   "relations": [{"name": "holding", "object": "o1"}],
                   "x": 0, "y": 0, "w": 10, "h": 10},
        }
    )
    triples = scene_graph_to_triples(sg)
    assert "umbrella [black]" in triples
    assert "person --holding--> umbrella" in triples
    # Bare object with no attributes still gets its own line.
    assert "person" in triples


def test_scene_graph_to_triples_relation_to_unknown_object_falls_back_to_id():
    sg = _sg(
        {
            "p1": {"name": "person", "attributes": [],
                   "relations": [{"name": "holding", "object": "missing_id"}]},
        }
    )
    triples = scene_graph_to_triples(sg)
    assert "person --holding--> missing_id" in triples


def test_boxes_to_draw_only_connected_filters_bare_objects():
    sg = _sg(
        {
            "connected": {"name": "umbrella", "attributes": ["black"], "relations": [],
                          "x": 0, "y": 0, "w": 10, "h": 10},
            "bare": {"name": "sky", "attributes": [], "relations": [],
                     "x": 0, "y": 0, "w": 500, "h": 500},
        }
    )
    kept_connected_only = {oid for oid, _ in _boxes_to_draw(sg, only_connected=True)}
    assert kept_connected_only == {"connected"}

    kept_all = {oid for oid, _ in _boxes_to_draw(sg, only_connected=False)}
    assert kept_all == {"connected", "bare"}


def test_boxes_to_draw_min_area_filters_tiny_boxes():
    sg = _sg(
        {
            "tiny": {"name": "speck", "attributes": ["red"], "relations": [],
                     "x": 0, "y": 0, "w": 2, "h": 2},
            "big": {"name": "car", "attributes": ["red"], "relations": [],
                    "x": 0, "y": 0, "w": 100, "h": 50},
        }
    )
    kept = {oid for oid, _ in _boxes_to_draw(sg, only_connected=False, min_area=100.0)}
    assert kept == {"big"}


def test_boxes_to_draw_skips_objects_missing_box_fields():
    sg = _sg(
        {
            "no_box": {"name": "ghost", "attributes": ["red"], "relations": []},
            "has_box": {"name": "car", "attributes": ["red"], "relations": [],
                        "x": 0, "y": 0, "w": 10, "h": 10},
        }
    )
    kept = {oid for oid, _ in _boxes_to_draw(sg, only_connected=False)}
    assert kept == {"has_box"}


def test_draw_boxes_returns_axes_and_respects_only_connected(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    from PIL import Image

    from grounded_memory.data.review import draw_boxes

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (50, 50), color="white").save(image_path)

    sg = _sg(
        {
            "connected": {"name": "umbrella", "attributes": ["black"], "relations": [],
                          "x": 1, "y": 1, "w": 10, "h": 10},
            "bare": {"name": "sky", "attributes": [], "relations": [],
                     "x": 0, "y": 0, "w": 40, "h": 40},
        }
    )

    ax = draw_boxes(image_path, sg, only_connected=True)
    assert len(ax.patches) == 1  # only the connected object's box

    ax_all = draw_boxes(image_path, sg, only_connected=False)
    assert len(ax_all.patches) == 2
    plt.close("all")


def test_manifest_roundtrip_same_columns_and_order(tmp_path):
    rows = [
        ManifestRow(
            image_id="1", split="val", primary_bucket="A_multi_object",
            num_objects=5, num_relations=3, num_attributes=2, selection_reason="reason",
            scene_graph_id="1", manually_reviewed=True, ambiguity_notes="looks fine",
            expected_question_types="at least 5 meaningful objects",
        ),
        ManifestRow(
            image_id="2", split="val", primary_bucket="B_attributes",
            num_objects=4, num_relations=1, num_attributes=6, selection_reason="reason2",
            scene_graph_id="2", manually_reviewed=False,
        ),
    ]
    out_path = tmp_path / "manifest.review.csv"
    save_manifest(rows, out_path)

    header_line = out_path.read_text().splitlines()[0]
    assert header_line.split(",") == ManifestRow.csv_header()

    loaded = load_manifest(out_path)
    assert len(loaded) == len(rows)
    assert [r.to_csv_row() for r in loaded] == [r.to_csv_row() for r in rows]
    assert loaded[0].manually_reviewed is True
    assert loaded[1].manually_reviewed is False


def test_append_rejection_writes_header_once_and_appends(tmp_path):
    log_path = tmp_path / "gqa_review_log.csv"
    append_rejection(log_path, "img1", "C_spatial", "bad box", replaced_by="img2")
    append_rejection(log_path, "img2", "C_spatial", "still bad", replaced_by="img3")

    lines = log_path.read_text().splitlines()
    assert lines[0] == "image_id,bucket,reason,replaced_by,timestamp"
    assert len(lines) == 3  # header + 2 rows, never overwritten

    log = load_rejection_log(log_path)
    assert [row["image_id"] for row in log] == ["img1", "img2"]
    assert log[0]["replaced_by"] == "img2"
    assert load_rejected_ids(log_path) == {"img1", "img2"}


def test_load_rejected_ids_empty_when_log_missing(tmp_path):
    assert load_rejected_ids(tmp_path / "does_not_exist.csv") == set()


def test_reject_and_replace_swaps_in_next_best_candidate(tmp_path):
    scene_graphs = {
        "used1": _spatial_sg("used1", 1),
        "candidate_hi": _spatial_sg("candidate_hi", 3),
        "candidate_lo": _spatial_sg("candidate_lo", 2),
    }
    rows = [_manifest_row("used1")]
    log_path = tmp_path / "gqa_review_log.csv"

    warning = reject_and_replace(rows, 0, scene_graphs, "occluded", log_path)

    assert warning is None
    assert rows[0].image_id == "candidate_hi"  # best-scoring remaining candidate
    assert rows[0].primary_bucket == "C_spatial"
    assert rows[0].manually_reviewed is False  # replacement needs its own review
    assert "used1" in rows[0].selection_reason
    assert "occluded" in rows[0].selection_reason

    log = load_rejection_log(log_path)
    assert len(log) == 1
    assert log[0] == {
        "image_id": "used1", "bucket": "C_spatial", "reason": "occluded",
        "replaced_by": "candidate_hi", "timestamp": log[0]["timestamp"],
    }


def test_reject_and_replace_excludes_previously_rejected_images(tmp_path):
    scene_graphs = {
        "used1": _spatial_sg("used1", 1),
        "candidate_hi": _spatial_sg("candidate_hi", 3),
        "candidate_lo": _spatial_sg("candidate_lo", 2),
    }
    rows = [_manifest_row("used1")]
    log_path = tmp_path / "gqa_review_log.csv"

    reject_and_replace(rows, 0, scene_graphs, "r1", log_path)  # used1 -> candidate_hi
    reject_and_replace(rows, 0, scene_graphs, "r2", log_path)  # candidate_hi -> ?

    # used1 was already rejected once -- must not come back even though it's
    # no longer "in use" in `rows`.
    assert rows[0].image_id == "candidate_lo"
    assert {r["image_id"] for r in load_rejection_log(log_path)} == {"used1", "candidate_hi"}


def test_reject_and_replace_pool_exhausted_removes_row_and_warns(tmp_path):
    scene_graphs = {"only_used": _spatial_sg("only_used", 1)}
    rows = [_manifest_row("only_used")]
    log_path = tmp_path / "gqa_review_log.csv"

    warning = reject_and_replace(rows, 0, scene_graphs, "bad", log_path)

    assert warning is not None
    assert "exhausted" in warning
    assert rows == []  # removed, not silently left in place with stale data

    log = load_rejection_log(log_path)
    assert log[0]["image_id"] == "only_used"
    assert log[0]["replaced_by"] == ""
