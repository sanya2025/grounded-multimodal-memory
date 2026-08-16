#!/bin/bash
# =============================================================================
# Experiment 1 — Grounded Scene Understanding, as a Slurm ARRAY job.
#
# 2 models x 3 prompt conditions = 6 independent tasks. Each array index runs one
# (model, condition) pair on its own GPU, so the whole 120 x 2 x 3 = 720-prediction
# run fans out across up to 6 GPUs at once.
#
# Submit:   sbatch slurm/e1_array.sbatch
# Watch:    squeue -u $USER        |  tail -f slurm/logs/e1_<jobid>_<idx>.out
#
# EDIT BEFORE RUNNING:
#   - CONDA_ENV / activation line (Section: environment)
#   - GQA_IMAGES path (point at your dataset on scratch)
#   - Confirm partition name (-p gpu) and GPU type match `sinfo`
#   - Confirm model names match configs/models.yaml
# =============================================================================
#SBATCH --job-name=gmm-e1
#SBATCH --partition=gpu              # confirm with: sinfo -s
#SBATCH --gres=gpu:1                 # one GPU per task; use gpu:a100:1 to pin a type
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5                  # 6 tasks: 2 models x 3 conditions
#SBATCH --output=slurm/logs/e1_%A_%a.out
#SBATCH --error=slurm/logs/e1_%A_%a.err

set -euo pipefail

# ---- environment ------------------------------------------------------------
module load python3
# Activate your conda env (created once — see HPC_SJSU.md, "Environment"):
source "$HOME/miniconda3/bin/activate" gmm

# Keep model weights and HF cache OFF your home quota:
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export GMM_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false

# ---- dataset (EDIT this path) ----------------------------------------------
GQA_IMAGES="${SCRATCH:-$HOME}/gqa/images"

# ---- map array index -> (model, condition) ---------------------------------
# Keep these two arrays in sync with configs/models.yaml + experiments.yaml.
MODELS=(qwen2_5_vl_7b llava_next_7b)
CONDITIONS=(standard grounded structured)

n_cond=${#CONDITIONS[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID / n_cond ))
cond_idx=$(( SLURM_ARRAY_TASK_ID % n_cond ))
MODEL="${MODELS[$model_idx]}"
CONDITION="${CONDITIONS[$cond_idx]}"

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} condition=${CONDITION} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- run one (model, condition) --------------------------------------------
srun python scripts/run_scene_experiment.py \
    --model "${MODEL}" \
    --condition "${CONDITION}" \
    --images-dir "${GQA_IMAGES}"

echo "[$(date)] done task ${SLURM_ARRAY_TASK_ID}."
# After ALL array tasks finish, compute metrics once (locally or via slurm/e1_eval.sbatch):
#   python scripts/evaluate_predictions.py
