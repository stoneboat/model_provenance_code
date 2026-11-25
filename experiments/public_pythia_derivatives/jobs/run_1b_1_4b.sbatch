#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --job-name=pythia_audit_1b_1_4b
#SBATCH --output=experiments/public_pythia_derivatives/logs/%x_%j.out
#SBATCH --error=experiments/public_pythia_derivatives/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
test -x "$ENV_DIR/bin/python"
conda activate "$ENV_DIR"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

python experiments/public_pythia_derivatives/runners/run_audit.py \
  --families 1b,1.4b \
  --seeds "${PYTHIA_SEEDS:-0}" \
  --batch-size "${PYTHIA_BATCH_SIZE:-4}" \
  --dtype "${PYTHIA_DTYPE:-float16}" \
  --skip-prepare --skip-unavailable \
  --resume
