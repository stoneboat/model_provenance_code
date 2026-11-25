#!/usr/bin/env bash
# Run after all three final targets have been atomically published.
set -euo pipefail
cd "$(dirname "$0")/../../.."
run=artifacts/m7vv_q4
p=.cache/vit-venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
mkdir -p "$run/audit"
# Set this for a reproduction in a different wall-clock session.
deadline=${M7VV_GPU_DEADLINE_UTC:-"2026-11-25 01:23:00 UTC"}
cpu_jobs=()
for target in A B C; do
    checkpoint="$run/train/$target/checkpoints/target_epoch1"
    args=(--run-dir "$run" --checkpoint-name "$target" --checkpoint-path "$checkpoint")
    # Wall-clock guard for this experiment includes any time already used by generation and training.
    remaining=$(( $(date -ud "$deadline" +%s) - $(date -u +%s) ))
    test "$remaining" -gt 0
    timeout --kill-after=4s "${remaining}s" "$p" -m experiments.watermark_inheritance.score "${args[@]}" --predictions-only > "$run/audit/$target-predict.log" 2>&1
    "$p" -m experiments.watermark_inheritance.score "${args[@]}" --cpu-only > "$run/audit/$target-score.log" 2>&1 &
    cpu_jobs+=("$!")
done
code=0
for pid in "${cpu_jobs[@]}"; do wait "$pid" || code=1; done
test "$code" -eq 0
"$p" -m experiments.watermark_inheritance.matched_process_audit --run-dir "$run" --output-dir "$run/audit"
