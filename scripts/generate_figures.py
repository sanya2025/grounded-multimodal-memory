#!/usr/bin/env python
"""Generate blog figures from FROZEN result files only (never invented numbers).

Reads result JSON/CSV under results/ and writes figures to results/figures/.
Requires the 'viz' extra (matplotlib). If no frozen results exist yet, prints the
list of expected figures rather than fabricating data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_FIGURES = [
    # Blog 1
    "fig01_architecture", "fig04_object_relation_spatial", "fig05_hallucination_rate",
    "fig06_evidence_support", "fig07_failure_taxonomy",
    # Blog 2
    "fig12_recall_at_k", "fig13_mrr_ndcg", "fig14_memory_ablation", "fig16_topk_sweep",
    # Blog 3
    "fig19_latency_breakdown", "fig20_accuracy_vs_latency", "fig22_memory_vs_accuracy",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures from frozen results.")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    results = Path(args.results_dir)
    tables = results / "tables"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    e2 = tables / "e2_retrieval_summary.json"
    made = 0
    if e2.exists():
        from grounded_memory.plotting.figures import grouped_bar

        data = json.loads(e2.read_text())
        methods = list(data.keys())
        recall = [data[m].get("recall@5", 0.0) for m in methods]
        mrr = [data[m].get("mrr", 0.0) for m in methods]
        grouped_bar(
            methods, {"Recall@5": recall, "MRR": mrr},
            ylabel="score", title="E2.1 retrieval quality by method",
            out_path=figures / "fig12_recall_at_k.png",
        )
        made += 1
        print(f"Generated fig12_recall_at_k.png from {e2}")

    if made == 0:
        print("No frozen result tables found yet. Expected figures once results exist:")
        for name in EXPECTED_FIGURES:
            print(f"  - {name}.png")
        print("\nRun the experiments and freeze result tables first (do NOT invent numbers).")


if __name__ == "__main__":
    main()
