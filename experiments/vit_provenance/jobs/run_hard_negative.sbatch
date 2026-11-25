#!/bin/bash
# AISTATS Reviewer 3 Q2: independent ImageNet-1K transformer hard negative.
# Submit smoke first:
#   sbatch --export=ALL,HARD_NEGATIVE_MODE=smoke experiments/vit_provenance/jobs/run_hard_negative.sbatch
# Then submit production after smoke passes:
#   sbatch experiments/vit_provenance/jobs/run_hard_negative.sbatch
#SBATCH --job-name=vit-hard-neg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/vit_provenance/logs/%x_%j.out
#SBATCH --error=experiments/vit_provenance/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

module purge
module load pytorch/25 anaconda3
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  bash scripts/cluster/install_pace.sh
fi
conda activate "$ENV_DIR"
PYTHON="$ENV_DIR/bin/python"
test -x "$PYTHON"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

MODE="${HARD_NEGATIVE_MODE:-production}"
EXTRA=()
if [ "$MODE" = "smoke" ]; then
  EXTRA+=(--smoke --smoke-per-label "${HARD_NEGATIVE_SMOKE_PER_LABEL:-8}")
elif [ "$MODE" != "production" ]; then
  echo "HARD_NEGATIVE_MODE must be smoke or production; got $MODE" >&2
  exit 2
fi

"$PYTHON" experiments/vit_provenance/runners/run_hard_negative.py \
  --batch-size "${VIT_BATCH_SIZE:-16}" \
  --dtype float16 \
  --device cuda \
  --resume \
  "${EXTRA[@]}"
