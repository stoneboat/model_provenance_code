#!/bin/bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p experiments/public_pythia_derivatives/logs

gpu_options=()
cpu_options=()
if [ -n "${PYTHIA_ACCOUNT:-}" ]; then
  gpu_options+=(--account="$PYTHIA_ACCOUNT")
  cpu_options+=(--account="$PYTHIA_ACCOUNT")
fi
if [ -n "${PYTHIA_GPU_PARTITION:-}" ]; then
  gpu_options+=(--partition="$PYTHIA_GPU_PARTITION")
fi
if [ -n "${PYTHIA_CPU_PARTITION:-}" ]; then
  cpu_options+=(--partition="$PYTHIA_CPU_PARTITION")
fi

job_prepare="$(sbatch --parsable "${cpu_options[@]}" experiments/public_pythia_derivatives/jobs/prepare.sbatch)"
job_small="$(sbatch --parsable "${gpu_options[@]}" --dependency="afterok:${job_prepare}" experiments/public_pythia_derivatives/jobs/run_1b_1_4b.sbatch)"
job_medium="$(sbatch --parsable "${gpu_options[@]}" --dependency="afterok:${job_prepare}" experiments/public_pythia_derivatives/jobs/run_6_9b.sbatch)"
job_large="$(sbatch --parsable "${gpu_options[@]}" --dependency="afterok:${job_prepare}" experiments/public_pythia_derivatives/jobs/run_12b.sbatch)"
job_report="$(sbatch --parsable "${cpu_options[@]}" --dependency="afterok:${job_small}:${job_medium}:${job_large}" experiments/public_pythia_derivatives/jobs/compile.sbatch)"

printf 'prepare: %s\n1B/1.4B: %s\n6.9B: %s\n12B: %s\nreport: %s\n' \
  "$job_prepare" "$job_small" "$job_medium" "$job_large" "$job_report"
