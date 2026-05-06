#!/bin/bash
set -euo pipefail

module purge
module load pytorch/25
module load anaconda3

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${ENV_NAME:-model_provenance_venv}"
ENV_DIR="${ENV_DIR:-/tmp/python-venv/$ENV_NAME}"

if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="$REPO_ROOT/data/.hf_home"
fi
mkdir -p "$HF_HOME"

mkdir -p /tmp/python-venv

if [ -d "$ENV_DIR" ]; then
    echo "Conda env already exists at $ENV_DIR"
else
    echo "Creating conda env at $ENV_DIR"
    conda create --prefix "$ENV_DIR" python=3.11 -y
fi

conda activate "$ENV_DIR"

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
hash -r

"$PIP" install --upgrade pip
PYTHONNOUSERSITE=1 "$PIP" install -r "$REPO_ROOT/requirements.txt"
if [ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/setup.py" ]; then
    PYTHONNOUSERSITE=1 "$PIP" install -e "$REPO_ROOT"
fi

mkdir -p "$REPO_ROOT/data/outputs"
mkdir -p "$REPO_ROOT/data/models"
mkdir -p "$REPO_ROOT/data/datasets"
mkdir -p "$REPO_ROOT/data/.hf_home"

cat <<EOF
Installation complete.
Activate with:
  conda activate $ENV_DIR

Example run:
  bash $REPO_ROOT/scripts/local_scripts/run_pace.sh \\
    --input_path /path/to/data.csv \\
    --model_name_or_path /path/to/local/model \\
    --dataset_desc_file /path/to/dataset_desc.txt \\
    --hf_model_id google/gemma-3-1b-it
EOF
