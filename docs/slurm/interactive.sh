#!/bin/bash
# =============================================================================
# Grab an INTERACTIVE GPU session on the SJSU CoE HPC.
#
# Use this for development, smoke tests, and debugging BEFORE submitting batch
# jobs. It drops you into a shell on a GPU compute node with the project env
# ready. Nothing here runs long work — do that with sbatch (e1_array / e2).
#
# Usage:
#   bash slurm/interactive.sh              # 1 GPU, 2h, gpu partition
#   bash slurm/interactive.sh a100 4:00:00 # request an A100 for 4 hours
#
# Then, inside the session:
#   python scripts/run_scene_experiment.py --images-dir $SCRATCH/gqa/images --limit 5
# =============================================================================
set -euo pipefail

GPU_TYPE="${1:-}"          # e.g. a100 | h100 | v100 | p100 ; empty = any
WALLTIME="${2:-02:00:00}"
CPUS="${3:-8}"
MEM="${4:-32G}"

if [[ -n "${GPU_TYPE}" ]]; then
    GRES="gpu:${GPU_TYPE}:1"
else
    GRES="gpu:1"
fi

echo "Requesting interactive session: --gres=${GRES} --time=${WALLTIME} ..."
echo "(Once on the node, run:  module load python3 && source \$HOME/miniconda3/bin/activate gmm)"

exec srun \
    --partition=gpu \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${WALLTIME}" \
    --pty bash
