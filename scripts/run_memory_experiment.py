#!/usr/bin/env python
"""Run Experiment 2 (E2.1-E2.5): temporal memory & retrieval comparison.

Loads temporal sequences + questions, builds a memory store, and evaluates the
four methods (no_memory, full_history, semantic, hybrid) plus ablations. Records
retrieval rankings and answers so retrieval quality and answer quality are scored
SEPARATELY downstream.

Use --mock (default when no models installed) to run offline with the hashing
embedder + MockVLM. Replace with a real encoder/VLM for actual experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_memory.config import load_config
from grounded_memory.evaluation.retrieval import mean_recall_at_k, mrr
from grounded_memory.memory.embeddings import HashingTextEmbedder
from grounded_memory.memory.events import MemoryEvent
from grounded_memory.memory.store import MemoryStore
from grounded_memory.pipelines.memory_qa import retrieve_for_question


def load_sequences(sequences_dir: Path) -> list[dict]:
    """Load JSON sequence files: {events:[...], questions:[...]}."""
    files = sorted(sequences_dir.glob("*.json"))
    return [json.loads(f.read_text()) for f in files]


def build_store(seq: dict, embedder: HashingTextEmbedder) -> MemoryStore:
    from datetime import datetime

    store = MemoryStore()
    for ev in seq["events"]:
        ts = ev["timestamp"]
        store.add(
            MemoryEvent(
                event_id=ev["event_id"],
                timestamp=datetime.fromisoformat(ts) if isinstance(ts, str) else ts,
                image_path=ev.get("image_path", ""),
                description=ev["description"],
                entities=ev.get("entities", []),
                embedding=embedder.embed_text(ev["description"]),
                sequence_id=seq.get("sequence_id"),
            )
        )
    return store


def main() -> None:
    exp_cfg = load_config("experiments")["e2_multimodal_memory"]
    ap = argparse.ArgumentParser(description="Run E2 memory & retrieval experiment.")
    ap.add_argument("--sequences-dir", default=exp_cfg["sequences_dir"])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    seq_dir = Path(args.sequences_dir)
    sequences = load_sequences(seq_dir) if seq_dir.exists() else []
    if not sequences:
        print(f"[warn] no sequence files in {seq_dir}. See notebook 08 to build them.")
        return

    embedder = HashingTextEmbedder()
    summary: dict[str, dict[str, float]] = {}
    for method in exp_cfg["methods"]:
        retrieved_lists, relevant_sets = [], []
        for seq in sequences:
            store = build_store(seq, embedder)
            for q in seq["questions"]:
                out = retrieve_for_question(
                    method, store,
                    query_embedding=embedder.embed_text(q["question"]),
                    query_entities=set(q.get("entities", [])),
                    top_k=args.top_k,
                )
                retrieved_lists.append(out.retrieved_ids)
                relevant_sets.append(set(q.get("relevant_event_ids", [])))
        summary[method] = {
            "recall@5": round(mean_recall_at_k(retrieved_lists, relevant_sets, 5), 4),
            "mrr": round(mrr(retrieved_lists, relevant_sets), 4),
            "n_questions": len(retrieved_lists),
        }

    print(json.dumps(summary, indent=2))
    out_path = Path("results/tables/e2_retrieval_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}. NOTE: retrieval quality != answer quality; run QA too.")


if __name__ == "__main__":
    main()
