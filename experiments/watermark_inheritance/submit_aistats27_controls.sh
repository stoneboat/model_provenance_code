#!/bin/bash
set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
ACCOUNT=()
if [[ -n "${WM_SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT=(--account="$WM_SLURM_ACCOUNT")
fi
GPU_PARTITION=${WM_GPU_PARTITION:-gpu-h200}
CPU_PARTITION=${WM_CPU_PARTITION:-cpu-small}
"$PYTHON" -m experiments.watermark_inheritance.reviewer_configure --root "$ROOT"

STAGE=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$CPU_PARTITION" \
  experiments/watermark_inheritance/jobs/aistats27_stage_inputs.sbatch)
PILOT=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$GPU_PARTITION" \
  --dependency="afterok:$STAGE" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_pilot.sbatch)
DOLLY=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PILOT" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_dolly.sbatch)
SHARDS=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PILOT" \
  experiments/watermark_inheritance/jobs/aistats27_ultrachat_shards.sbatch)
PREP=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$CPU_PARTITION" \
  --dependency="afterok:$SHARDS" \
  experiments/watermark_inheritance/jobs/aistats27_ultrachat_prepare.sbatch)
ULTRA=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PREP" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_ultrachat.sbatch)
REPORT=$(sbatch --parsable "${ACCOUNT[@]}" --partition="$CPU_PARTITION" \
  --dependency="afterok:$DOLLY:$ULTRA" \
  experiments/watermark_inheritance/jobs/aistats27_negative_control_report.sbatch)
printf 'Inputs %s, pilot %s, Dolly %s, UltraChat shards %s, UltraChat preparation %s, UltraChat %s, report %s\n' \
  "$STAGE" "$PILOT" "$DOLLY" "$SHARDS" "$PREP" "$ULTRA" "$REPORT"
