#!/usr/bin/env python
"""Run Experiment 1 (E1.1-E1.3): 120 images x 2 VLMs x 3 prompt conditions.

Writes raw prediction JSONL per (model, condition) under results/predictions/.
Metrics are computed separately (scripts/evaluate_predictions.py) from the frozen
raw predictions.

Requires real models (pip install -e '.[models]') and, in practice, a GPU. Use
--mock to exercise the full pipeline offline with the deterministic MockVLM.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grounded_memory.config import load_config
from grounded_memory.models import load_model
from grounded_memory.models.mock import MockVLM
from grounded_memory.pipelines.scene_understanding import run_conditions_for_image
from grounded_memory.tracking import make_run_id, write_jsonl


def load_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    exp_cfg = load_config("experiments")
    data_cfg = load_config("datasets")
    e1 = exp_cfg["e1_grounded_scene_understanding"]
    ap = argparse.ArgumentParser(description="Run E1 grounded scene understanding.")
    ap.add_argument("--mock", action="store_true", help="Use MockVLM (offline).")
    ap.add_argument("--manifest", default=data_cfg["manifest"]["path"])
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Limit #images (smoke test).")
    ap.add_argument(
        "--model", default=None,
        help="Run only this model from configs/models.yaml (default: all E1 models). "
        "Lets a Slurm array task target one model.",
    )
    ap.add_argument(
        "--condition", default=None, choices=e1["prompt_conditions"],
        help="Run only this prompt condition (default: all). Lets an array task "
        "target one condition.",
    )
    args = ap.parse_args()

    if args.mock:
        model_names = ["mock"]
    elif args.model:
        model_names = [args.model]
    else:
        model_names = e1["models"]
    conditions = [args.condition] if args.condition else e1["prompt_conditions"]
    run_id = make_run_id("e1")

    manifest = load_manifest(args.manifest) if Path(args.manifest).exists() else []
    if not manifest:
        print(f"[warn] manifest {args.manifest} missing/empty; running one synthetic image.")
        manifest = [{"image_id": "demo_0", "primary_bucket": "A_multi_object"}]
    if args.limit:
        manifest = manifest[: args.limit]

    images_root = Path(args.images_dir) if args.images_dir else None
    for name in model_names:
        model = MockVLM() if name == "mock" else load_model(name)
        for condition in conditions:
            records = []
            for row in manifest:
                image_id = row["image_id"]
                image = None
                if images_root is not None:
                    image = str(images_root / f"{image_id}.jpg")
                recs = run_conditions_for_image(
                    model, image, sample_id=image_id,
                    experiment_id="E1", run_id=run_id, conditions=[condition],
                    dataset="gqa", dataset_version=data_cfg["manifest"]["version"],
                    manifest_version=data_cfg["manifest"]["version"],
                )
                records.extend(recs)
            out = Path(e1["results_dir"]) / f"e1_{model.name}_{condition}__{run_id}.jsonl"
            write_jsonl(records, out, overwrite=True)
            print(f"wrote {len(records):4d} predictions -> {out}")
    print(f"\nDone. run_id={run_id}. Next: scripts/evaluate_predictions.py")


if __name__ == "__main__":
    main()
