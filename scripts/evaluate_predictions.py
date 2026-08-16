#!/usr/bin/env python
"""Compute deterministic metrics from frozen raw prediction JSONL files.

Keeps raw predictions and derived metrics separate (Reproducibility rule #7).
This is a scaffold: it loads predictions, reports parser-failure accounting, and
leaves TODO hooks where scene-graph references are joined for object/relation/
spatial/hallucination metrics (those require the GQA scene graphs on disk).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate frozen predictions.")
    ap.add_argument("--predictions-dir", default="results/predictions")
    ap.add_argument("--glob", default="e1_*.jsonl")
    args = ap.parse_args()

    files = sorted(Path(args.predictions_dir).glob(args.glob))
    if not files:
        print(f"No prediction files matching {args.glob} in {args.predictions_dir}.")
        return

    per_group: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "parse_ok": 0, "parse_fail": 0}
    )
    for f in files:
        for rec in load_jsonl(f):
            condition = rec.get("metrics", {}).get("condition", "?")
            key = (rec.get("model_id", "?"), condition)
            g = per_group[key]
            g["n"] += 1
            if condition == "structured":
                if rec.get("parse_ok"):
                    g["parse_ok"] += 1
                else:
                    g["parse_fail"] += 1

    print("Parser-failure accounting (structured condition):")
    print(f"{'model':40s} {'condition':12s} {'n':>5s} {'parse_ok':>9s} {'parse_fail':>10s}")
    for (model, condition), g in sorted(per_group.items()):
        print(f"{model:40s} {condition:12s} {g['n']:5d} {g['parse_ok']:9d} {g['parse_fail']:10d}")

    print(
        "\nTODO: join each prediction to its GQA scene graph and compute "
        "object/attribute/relation/spatial + tri-state hallucination metrics with "
        "bootstrap CIs (see src/grounded_memory/evaluation/). Metrics on real model "
        "outputs require the GQA scene graphs on disk."
    )


if __name__ == "__main__":
    main()
