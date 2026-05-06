#!/usr/bin/env bash
# Phase 1 local environment setup wrapper.
# On PACE/HPC: run scripts/local_scripts/install_pace.sh instead for
# full conda env creation with module loading.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[setup] repo root: $REPO_ROOT"
echo "[setup] python:    $(python --version 2>&1 || echo 'not found')"

# ------------------------------------------------------------------ #
# Existing HPC setup script: call if present and we're on a system
# that has the 'module' command.  Skip silently otherwise.
# ------------------------------------------------------------------ #
PACE_SCRIPT="$REPO_ROOT/scripts/local_scripts/install_pace.sh"
if [[ -f "$PACE_SCRIPT" ]] && command -v module >/dev/null 2>&1; then
    echo "[setup] HPC 'module' command detected — delegating to $PACE_SCRIPT"
    bash "$PACE_SCRIPT"
    echo "[setup] install_pace.sh finished."
else
    echo "[setup] HPC module system not available; running lightweight local setup."

    # Install Python dependencies
    python -m pip install --upgrade pip --quiet
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        echo "[setup] Installing requirements.txt ..."
        python -m pip install -r "$REPO_ROOT/requirements.txt" --quiet
    else
        echo "[setup] No requirements.txt found — skipping pip install."
    fi

    # Install package in editable mode if pyproject.toml present
    if [[ -f "$REPO_ROOT/pyproject.toml" ]] || [[ -f "$REPO_ROOT/setup.py" ]]; then
        echo "[setup] Installing package in editable mode ..."
        python -m pip install -e "$REPO_ROOT" --quiet
    fi
fi

# ------------------------------------------------------------------ #
# Ensure required directories exist (idempotent)
# ------------------------------------------------------------------ #
echo "[setup] Creating required directories ..."
mkdir -p \
    "$REPO_ROOT/data/raw" \
    "$REPO_ROOT/data/processed" \
    "$REPO_ROOT/data/scores" \
    "$REPO_ROOT/outputs/runs" \
    "$REPO_ROOT/outputs/reports" \
    "$REPO_ROOT/outputs/figures"

# ------------------------------------------------------------------ #
# Set HF_HOME if not already set
# ------------------------------------------------------------------ #
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="$REPO_ROOT/data/.hf_home"
    mkdir -p "$HF_HOME"
    echo "[setup] HF_HOME set to $HF_HOME"
fi

# ------------------------------------------------------------------ #
# Run environment checker
# ------------------------------------------------------------------ #
echo ""
echo "[setup] Running environment check ..."
python "$REPO_ROOT/scripts/setup/check_env.py"
