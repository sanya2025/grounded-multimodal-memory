#!/usr/bin/env python
"""Build the CANDIDATE 120-image GQA manifest for manual review.

Reads GQA scene graphs, scores each image per bucket, greedily fills six 20-image
buckets, and writes a CSV with a `manually_reviewed` column set to False.

IMPORTANT: This produces a *candidate* manifest. A human must inspect the images
and set manually_reviewed=True (and fix ambiguity notes / bucket assignments)
before the manifest is frozen as v1. Do not run experiments on an unreviewed
manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grounded_memory.config import load_config
from grounded_memory.data.gqa import load_scene_graphs, scene_graph_path
from grounded_memory.data.selection import ManifestRow, build_candidate_manifest


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ManifestRow.csv_header())
        for row in rows:
            writer.writerow(row.to_csv_row())


def main() -> None:
    cfg = load_config("datasets")
    ap = argparse.ArgumentParser(description="Build candidate GQA benchmark manifest.")
    ap.add_argument("--split", default=cfg["gqa"].get("benchmark_split", "val"))
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--out", default=cfg["manifest"]["path"])
    args = ap.parse_args()

    sg_path = scene_graph_path(cfg, split=args.split)
    print(f"Loading scene graphs from {sg_path} ...")
    scene_graphs = load_scene_graphs(sg_path)
    print(f"Loaded {len(scene_graphs)} scene graphs.")

    rows = build_candidate_manifest(scene_graphs, split=args.split, per_bucket=args.per_bucket)
    out = Path(args.out)
    write_manifest(rows, out)
    print(f"Wrote {len(rows)} candidate rows to {out}")
    print("NEXT: manually review each image, set manually_reviewed=True, then freeze as v1.")


if __name__ == "__main__":
    main()
