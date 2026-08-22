#!/usr/bin/env python
"""Compute deterministic metrics from frozen raw prediction JSONL files.

Keeps raw predictions and derived metrics separate (Reproducibility rule #7).
Parser-failure accounting covers all three prompt conditions. Object/attribute/
relation/spatial accuracy and tri-state hallucination/evidence-support metrics
are computed for the structured (JSON) condition only, joined against the GQA
scene graphs on disk -- standard/grounded predictions are free text with no
extraction path yet (see src/grounded_memory/evaluation/structured_scene.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from grounded_memory.config import load_config
from grounded_memory.data.gqa import SceneGraph, load_scene_graphs, scene_graph_path
from grounded_memory.evaluation.stats import bootstrap_ci
from grounded_memory.evaluation.structured_scene import score_structured_scene
from grounded_memory.scene.schema import SceneRepresentation


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _print_parse_table(per_group: dict[tuple[str, str], dict[str, int]]) -> None:
    print("Parser-failure accounting (structured condition):")
    print(f"{'model':40s} {'condition':12s} {'n':>5s} {'parse_ok':>9s} {'parse_fail':>10s}")
    for (model, condition), g in sorted(per_group.items()):
        print(f"{model:40s} {condition:12s} {g['n']:5d} {g['parse_ok']:9d} {g['parse_fail']:10d}")


def _score_structured_condition(
    records_by_group: dict[tuple[str, str], list[dict]],
    scene_graphs: dict[str, SceneGraph],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Per-(model, condition) lists of per-image metric values, structured condition only."""
    per_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing_scene_graph = 0
    for (model, condition), records in records_by_group.items():
        if condition != "structured":
            continue
        for rec in records:
            if not rec.get("parse_ok") or not rec.get("parsed_prediction"):
                continue
            sg = scene_graphs.get(rec["sample_id"])
            if sg is None:
                missing_scene_graph += 1
                continue
            scene = SceneRepresentation.model_validate(rec["parsed_prediction"])
            score = score_structured_scene(scene, sg)
            bucket = per_group[(model, condition)]
            for prf1_name, prf1 in (
                ("object", score.objects),
                ("attribute", score.attributes),
                ("relationship", score.relationships),
                ("spatial", score.spatial),
                ("positional", score.positional),
            ):
                bucket[f"{prf1_name}_precision"].append(prf1.precision)
                bucket[f"{prf1_name}_recall"].append(prf1.recall)
                bucket[f"{prf1_name}_f1"].append(prf1.f1)
            bucket["hallucination_rate"].append(score.hallucination_rate)
            bucket["evidence_support_rate"].append(score.evidence_support_rate)
    if missing_scene_graph:
        print(f"[warn] {missing_scene_graph} structured predictions had no matching scene graph.")
    return per_group


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate frozen predictions.")
    ap.add_argument("--predictions-dir", default="results/predictions")
    ap.add_argument("--glob", default="e1_*.jsonl")
    ap.add_argument("--tables-dir", default="results/tables")
    args = ap.parse_args()

    files = sorted(Path(args.predictions_dir).glob(args.glob))
    if not files:
        print(f"No prediction files matching {args.glob} in {args.predictions_dir}.")
        return

    per_group: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "parse_ok": 0, "parse_fail": 0}
    )
    records_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
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
            records_by_group[key].append(rec)

    _print_parse_table(per_group)

    try:
        datasets_cfg = load_config("datasets")
        scene_graphs = load_scene_graphs(scene_graph_path(datasets_cfg))
    except FileNotFoundError as e:
        print(f"\nSkipping structured-condition scene metrics -- need GQA first: {e}")
        return

    eval_cfg = load_config("evaluation")
    stats_cfg = eval_cfg.get("statistics", {})
    n_boot = stats_cfg.get("bootstrap_samples", 10000)
    ci = stats_cfg.get("ci", 0.95)
    seed = stats_cfg.get("seed", 20260815)

    scored = _score_structured_condition(records_by_group, scene_graphs)
    if not scored:
        print("\nNo structured-condition predictions with valid parsed JSON to score.")
        return

    print("\nStructured-condition scene metrics (mean [95% CI], bootstrapped over images):")
    print(
        "(relationship/spatial exclude GQA's exhaustive left/right positional\n"
        " annotations -- those are scored separately as 'positional'; see\n"
        " src/grounded_memory/evaluation/structured_scene.py)"
    )
    metric_names = [
        "object_precision", "object_recall", "object_f1",
        "attribute_precision", "attribute_recall", "attribute_f1",
        "relationship_precision", "relationship_recall", "relationship_f1",
        "spatial_precision", "spatial_recall", "spatial_f1",
        "positional_precision", "positional_recall", "positional_f1",
        "hallucination_rate", "evidence_support_rate",
    ]
    rows: list[dict[str, object]] = []
    for (model, condition), metrics in sorted(scored.items()):
        n = len(next(iter(metrics.values())))
        print(f"\n{model} / {condition}  (n={n} scored)")
        for name in metric_names:
            values = metrics[name]
            result = bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)
            print(f"  {name:22s} {result.point:.3f}  [{result.low:.3f}, {result.high:.3f}]")
            rows.append({
                "model_id": model, "condition": condition, "metric": name, "n": n,
                "mean": result.point, "ci_low": result.low, "ci_high": result.high,
            })

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "e1_structured_scene_metrics.csv"
    fieldnames = ["model_id", "condition", "metric", "n", "mean", "ci_low", "ci_high"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")

    print(
        "\nNote: standard/grounded conditions have no object/attribute/relation/"
        "spatial metrics -- there is no extraction path from free text yet. "
        "Also note hallucination_rate is currently always 0.0: the tri-state "
        "verifier (evaluation/grounding.py) never emits CONTRADICTED by design "
        "today, only SUPPORTED/NOT_VERIFIABLE -- real contradiction detection "
        "(e.g. model says 'red', scene graph says 'blue') is not implemented."
    )


if __name__ == "__main__":
    main()
