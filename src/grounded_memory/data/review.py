"""Manual-review helpers for the frozen 120-image GQA benchmark manifest.

Scene graphs are references, not omniscient truth -- a human must look at each
image before the manifest is frozen. This module is the pure/reusable half of
that review tool (notebook 02 is the thin UI on top of it). Importable with NO
datasets installed; matplotlib/PIL are only imported inside ``draw_boxes``, so
the base package stays light (see ADR 0001).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grounded_memory.data.gqa import SceneGraph
from grounded_memory.data.selection import BUCKETS, ManifestRow, next_candidate

REJECT_LOG_COLUMNS = ["image_id", "bucket", "reason", "replaced_by", "timestamp"]


def scene_graph_to_triples(sg: SceneGraph) -> list[str]:
    """Human-readable lines for one scene graph.

    One line per object (name, plus ``[attr1, attr2]`` if it has attributes),
    and one line per relation (``subject --relation--> object``), e.g.::

        umbrella [black]
        person --holding--> umbrella
    """
    lines: list[str] = []
    for obj_id, obj in sg.objects.items():
        name = obj.get("name", obj_id)
        attrs = obj.get("attributes") or []
        lines.append(f"{name} [{', '.join(attrs)}]" if attrs else name)
        for rel in obj.get("relations", []):
            target = sg.objects.get(rel.get("object", ""), {})
            target_name = target.get("name", rel.get("object", ""))
            lines.append(f"{name} --{rel.get('name', '')}--> {target_name}")
    return lines


def _boxes_to_draw(
    sg: SceneGraph, only_connected: bool = True, min_area: float = 0.0
) -> list[tuple[str, dict[str, Any]]]:
    """Filtering logic behind ``draw_boxes`` -- pure, testable without image I/O.

    ``only_connected`` skips objects with no attributes/relations, and
    ``min_area`` skips boxes below that pixel area; both cut GQA's usual
    clutter of tiny, unannotated background objects.
    """
    kept: list[tuple[str, dict[str, Any]]] = []
    for obj_id, obj in sg.objects.items():
        if not all(k in obj for k in ("x", "y", "w", "h")):
            continue
        if only_connected and not (obj.get("attributes") or obj.get("relations")):
            continue
        if float(obj["w"]) * float(obj["h"]) < min_area:
            continue
        kept.append((obj_id, obj))
    return kept


def draw_boxes(
    image_path: str | Path,
    sg: SceneGraph,
    only_connected: bool = True,
    min_area: float = 0.0,
    ax: Any | None = None,
) -> Any:
    """Draw GQA object boxes + name labels over the image; return the Axes.

    Boxes are GQA's pixel ``x, y, w, h`` (top-left + size). Requires the
    ``viz`` extra (matplotlib) -- imported lazily so importing this module
    doesn't require it. Deliberately does NOT force a backend (unlike
    plotting/figures.py's Agg-forcing): this is called from a live notebook
    kernel, which needs its own inline backend to actually display the
    figure -- forcing Agg here would silently suppress it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    img = Image.open(image_path)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(img)
    for obj_id, obj in _boxes_to_draw(sg, only_connected=only_connected, min_area=min_area):
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5))
        ax.text(
            x, max(y - 4, 0), obj.get("name", obj_id),
            color="white", fontsize=8, backgroundcolor="black", va="bottom",
        )
    ax.set_axis_off()
    ax.set_title(Path(image_path).name)
    return ax


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Load a manifest CSV, in ``ManifestRow.csv_header()``'s exact column order."""
    rows: list[ManifestRow] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                ManifestRow(
                    image_id=raw["image_id"],
                    split=raw["split"],
                    primary_bucket=raw["primary_bucket"],
                    num_objects=int(raw["num_objects"]),
                    num_relations=int(raw["num_relations"]),
                    num_attributes=int(raw["num_attributes"]),
                    selection_reason=raw["selection_reason"],
                    scene_graph_id=raw["scene_graph_id"],
                    source=raw["source"],
                    license_ref=raw["license_ref"],
                    manually_reviewed=raw["manually_reviewed"].strip().lower() == "true",
                    ambiguity_notes=raw["ambiguity_notes"],
                    expected_question_types=raw["expected_question_types"],
                )
            )
    return rows


def save_manifest(rows: list[ManifestRow], path: str | Path) -> None:
    """Write rows to CSV using ``ManifestRow.csv_header()``'s exact column order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ManifestRow.csv_header())
        for row in rows:
            writer.writerow(row.to_csv_row())


def append_rejection(
    path: str | Path, image_id: str, bucket: str, reason: str, replaced_by: str
) -> None:
    """Append one rejection record to the review log. Append-only, never overwrites.

    This is a SEPARATE file from the manifest (``ManifestRow.csv_header()`` stays
    the manifest's exact, un-drifted schema); the log is its own audit trail.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(REJECT_LOG_COLUMNS)
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        writer.writerow([image_id, bucket, reason, replaced_by, timestamp])


def load_rejection_log(path: str | Path) -> list[dict[str, str]]:
    """All rejection records from the append-only log, oldest first."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_rejected_ids(path: str | Path) -> set[str]:
    """image_ids that have ever been rejected, per the append-only log."""
    return {row["image_id"] for row in load_rejection_log(path)}


def reject_and_replace(
    rows: list[ManifestRow],
    idx: int,
    scene_graphs: dict[str, SceneGraph],
    reason: str,
    reject_log_path: str | Path,
) -> str | None:
    """Reject ``rows[idx]``, log it, and swap in the next-best same-bucket candidate.

    Mutates ``rows`` in place: on success, ``rows[idx]`` becomes a fresh,
    unreviewed ``ManifestRow`` for the replacement image, ready to be reviewed
    like any other row. If the bucket's candidate pool is exhausted, ``rows[idx]``
    is removed instead -- the bucket is left short by design, so the freeze
    check (every bucket at its target count) catches it rather than silently
    passing. Returns ``None`` on success, or a warning string when exhausted.
    """
    old_row = rows[idx]
    bucket = old_row.primary_bucket
    exclude = {r.image_id for r in rows} | load_rejected_ids(reject_log_path)

    new_id = next_candidate(scene_graphs, bucket, exclude)
    append_rejection(reject_log_path, old_row.image_id, bucket, reason, replaced_by=new_id or "")

    if new_id is None:
        del rows[idx]
        return (
            f"Bucket '{bucket}' candidate pool is exhausted -- no replacement found "
            f"for {old_row.image_id}. Bucket is now short one image; freeze will be "
            f"blocked until this is resolved."
        )

    sg = scene_graphs[new_id]
    rows[idx] = ManifestRow(
        image_id=new_id,
        split=old_row.split,
        primary_bucket=bucket,
        num_objects=sg.num_objects(),
        num_relations=sg.num_relations(),
        num_attributes=sg.num_attributes(),
        selection_reason=f"replacement for {old_row.image_id} (rejected: {reason})",
        scene_graph_id=new_id,
        source=old_row.source,
        license_ref=old_row.license_ref,
        manually_reviewed=False,
        expected_question_types=BUCKETS.get(bucket, ""),
    )
    return None


__all__ = [
    "scene_graph_to_triples", "draw_boxes", "load_manifest", "save_manifest",
    "append_rejection", "load_rejection_log", "load_rejected_ids", "reject_and_replace",
]
