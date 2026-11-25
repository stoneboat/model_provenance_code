#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
report=experiments/public_pythia_derivatives/results/2026-11-24-gpt-neo-lamini
out=artifacts/gpt_neo_lamini
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
py=.cache/vit-venv/bin/python
mkdir -p "$out/logs"
"$py" "$report/preflight.py" 2>&1 | tee "$out/logs/smoke.log"
"$py" experiments/public_pythia_derivatives/runners/train_matched_lamini.py --model gpt_neo --output-dir "$out/checkpoint" --protocol "$report/protocol.yaml" --data-dir artifacts/lamini_design 2>&1 | tee "$out/logs/training.log"
"$py" scripts/scoring/score_causal_lm_logprobs.py --model "$out/checkpoint/merged" --tokenizer-model EleutherAI/gpt-neo-1.3B --tokenizer-revision dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8 --train-file data/processed/mimir_github/train.jsonl --test-file data/processed/mimir_github/test.jsonl --device cuda:0 --dtype float16 --batch-size 4 --max-length 512 --min-k-pcts 20 --debug-examples 0 --output-dir "$out/mimir" 2>&1 | tee "$out/logs/scoring.log"
"$py" scripts/audit/run_threshold_audit.py --model-label GPT-Neo-Lamini --train-scores "$out/mimir/train_scores.jsonl" --test-scores "$out/mimir/test_scores.jsonl" --output-dir "$out/audit" --score-keys min_k_20_logprob,mean_logprob --primary-score min_k_20_logprob --criterion balanced_accuracy --n-thresholds 2000 --run-shuffled-control --shuffled-seed 42 2>&1 | tee "$out/logs/audit.log"
"$py" experiments/public_pythia_derivatives/reports/evaluate_lamini_adaptation.py --model gpt_neo --protocol "$report/protocol.yaml" --data-dir artifacts/lamini_design --merged "$out/checkpoint/merged" --output-dir "$out/downstream" 2>&1 | tee "$out/logs/downstream.log"
"$py" "$report/verify_results.py"
