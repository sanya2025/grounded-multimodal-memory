#!/bin/bash
# =============================================================================
# Experiment 2 — Multimodal Memory & Temporal Reasoning (single GPU job).
#
# Builds the memory store per sequence and runs the method comparison + retrieval
# metrics. E2 is far lighter than E1 (retrieval is cheap; a VLM/encoder is only
# needed if you swap in a real embedder), so one task on one GPU is plenty.
#
# Submit:   sbatch slurm/e2_memory.sbatch
#
# EDIT BEFORE RUNNING: conda activation line and SEQ_DIR if not the repo default.
# =============================================================================
#SBATCH --job-name=gmm-e2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/e2_%j.out
#SBATCH --error=slurm/logs/e2_%j.err

set -euo pipefail

module load python3
source "$HOME/miniconda3/bin/activate" gmm

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export GMM_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false

SEQ_DIR="data/sequences"     # your authored temporal sequences (see HPC_SJSU.md / runbook §8)

echo "[$(date)] E2 on $(hostname); sequences: ${SEQ_DIR}"

srun python scripts/run_memory_experiment.py \
    --sequences-dir "${SEQ_DIR}" \
    --top-k 5

echo "[$(date)] E2 done. Wrote results/tables/e2_retrieval_summary.json."
echo "Reminder: retrieval quality != answer quality — also score end-to-end QA."
