# Slurm scripts (SJSU CoE HPC)

Ready-to-run batch scripts for the SJSU College of Engineering cluster
(`coe-hpc.sjsu.edu`). Full walkthrough: [`../HPC_SJSU.md`](../HPC_SJSU.md).

| File | What it does | Submit |
|------|--------------|--------|
| `e1_array.sbatch` | Experiment 1 as a 6-task array (2 models × 3 conditions), one GPU per task | `sbatch slurm/e1_array.sbatch` |
| `e2_memory.sbatch` | Experiment 2 memory/retrieval comparison (single GPU) | `sbatch slurm/e2_memory.sbatch` |
| `interactive.sh` | Grab an interactive GPU node for dev/smoke tests | `bash slurm/interactive.sh` |
| `logs/` | sbatch stdout/stderr land here (`*_%A_%a.out`) | — |

## Before you submit — edit these

Every script has an **EDIT BEFORE RUNNING** block. At minimum:

1. **Conda activation line** — `source "$HOME/miniconda3/bin/activate" gmm`
   (change to your env name / path).
2. **Dataset path** — `GQA_IMAGES` / `SEQ_DIR` (point at your data on `$SCRATCH`).
3. **Partition & GPU** — confirm `--partition=gpu` and GPU type against
   `sinfo -s`. Pin a type with `--gres=gpu:a100:1` if you want a specific card.
4. **Model names** — the `MODELS=(...)` array in `e1_array.sbatch` must match
   `configs/models.yaml`.

## Typical flow

```bash
# 0. one-time: create the env (see ../HPC_SJSU.md)
# 1. dev / smoke test on a GPU
bash slurm/interactive.sh
#    (inside) python scripts/run_scene_experiment.py --images-dir $SCRATCH/gqa/images --limit 5

# 2. full E1 (fans out over 6 GPUs)
sbatch slurm/e1_array.sbatch
squeue -u $USER                      # watch

# 3. metrics once all array tasks finish
python scripts/evaluate_predictions.py

# 4. E2
sbatch slurm/e2_memory.sbatch
```

## Notes

- These request **1 GPU per task** — 7B VLMs in fp16 fit on a single A100/H100.
  Only add more GPUs if you shard a larger model.
- `%A` = array job id, `%a` = array task index, `%j` = job id (in log filenames).
- Keep the HF cache off your home quota: the scripts set
  `HF_HOME=$SCRATCH/hf_cache`.
- Cancel everything: `scancel -u $USER`. Cancel one array: `scancel <jobid>`.
