# Model Provenance — Shard Membership Auditing

This repository implements a membership inference pipeline to audit whether a
fine-tuned language model inherited pretraining membership signals from its
parent model. The core question: can we tell that a fine-tuned model was built
on top of a specific parent that was trained on a specific data shard?

The pipeline extracts per-token log-probabilities from causal LMs, computes
MIN-K% PROB scores, calibrates a threshold distinguisher on a held-out
calibration split, and evaluates shard advantage (TPR − FPR) on a test split.

**Parent model:** `EleutherAI/pythia-1.4b`  
**Target model:** `nnheui/pythia-1.4b-sft-full` (SFT on UltraChat 200k)  
**Dataset:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`, split=`ngram_13_0.2`)  
**Primary score:** `min_k_20_logprob`

---

## Requirements

- Python 3.11
- CUDA GPU (experiments run on Tesla V100-16GB)
- HuggingFace account with access to [`iamgroot42/mimir`](https://huggingface.co/datasets/iamgroot42/mimir)

---

## Installation

### On a PACE/HPC cluster

```bash
bash scripts/local_scripts/install_pace.sh
```

This creates a conda environment at `/tmp/python-venv/model_provenance_venv`.

### Activate the environment (every session)

```bash
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
```

### Set HuggingFace credentials

```bash
export HF_HOME=data/.hf_home
export HF_TOKEN=<your_hf_token>
```

### Verify the environment

```bash
python scripts/setup/check_env.py
```

---

## Repository Layout

```
scripts/
  data/
    prepare_mimir_github.py              # Prepare main member/nonmember dataset
    prepare_mimir_github_nonmember_control.py  # Prepare null-control dataset
  scoring/
    extract_logprob_scores.py            # Score a model on a prepared dataset
  experiments/
    run_mia_experiment.py                # Calibrate threshold and evaluate
  reports/
    compare_parent_target_advantage.py   # Parent vs target comparison report
    report_nonmember_control.py          # Null-control report

src/shard_audit/
  datasets.py        # MIMIR dataset loading
  preprocessing.py   # Text normalization and truncation
  splitting.py       # Stratified train/test splitting
  logprobs.py        # Per-token log-probability extraction
  mia_scores.py      # MIN-K% PROB and mean log-prob scoring
  distinguishers.py  # Threshold sweep, calibration, and evaluation
  sanity_checks.py   # Hash-overlap and balance checks
  metrics.py         # Score diagnostics (AUC, direction check)
```

---

## Reproducing the Experiments

### Experiment 1 — Parent model (EleutherAI/pythia-1.4b) on MIMIR GitHub

#### Step 1: Prepare the dataset

```bash
python scripts/data/prepare_mimir_github.py \
  --num-train-per-class 500 \
  --num-test-per-class 200 \
  --max-words 32 --min-words 8 \
  --seed 0 \
  --output-dir data/processed/mimir_github
```

Outputs `data/processed/mimir_github/train.jsonl` (1000 records) and
`test.jsonl` (400 records). Labels: `1` = member of Pythia's GitHub pretraining
corpus, `0` = nonmember.

#### Step 2: Score the parent model

```bash
python scripts/scoring/extract_logprob_scores.py \
  --model EleutherAI/pythia-1.4b \
  --train-file data/processed/mimir_github/train.jsonl \
  --test-file  data/processed/mimir_github/test.jsonl \
  --output-dir data/scores/mimir_github_pythia_mink \
  --min-k-pcts 5,10,20,40 \
  --batch-size 4
```

#### Step 3: Calibrate threshold and evaluate

```bash
python scripts/experiments/run_mia_experiment.py \
  --train-scores data/scores/mimir_github_pythia_mink/train_scores.jsonl \
  --test-scores  data/scores/mimir_github_pythia_mink/test_scores.jsonl \
  --output-dir   outputs/runs/mimir_github_pythia_mink \
  --primary-score min_k_20_logprob
```

---

### Experiment 2 — Target model (nnheui/pythia-1.4b-sft-full) on the same shard

Uses the same prepared dataset from Experiment 1, Step 1.

#### Step 1: Score the target model

```bash
python scripts/scoring/extract_logprob_scores.py \
  --model nnheui/pythia-1.4b-sft-full \
  --train-file data/processed/mimir_github/train.jsonl \
  --test-file  data/processed/mimir_github/test.jsonl \
  --output-dir data/scores/mimir_github_nnheui_pythia_1_4b_sft_full \
  --min-k-pcts 5,10,20,40 \
  --batch-size 4
```

#### Step 2: Calibrate threshold and evaluate (with controls)

```bash
python scripts/experiments/run_mia_experiment.py \
  --train-scores data/scores/mimir_github_nnheui_pythia_1_4b_sft_full/train_scores.jsonl \
  --test-scores  data/scores/mimir_github_nnheui_pythia_1_4b_sft_full/test_scores.jsonl \
  --output-dir   outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full \
  --primary-score min_k_20_logprob \
  --parent-results outputs/runs/mimir_github_pythia_mink/results.json \
  --run-shuffled-control
```

`--parent-results` enables the parent-threshold transfer diagnostic.  
`--run-shuffled-control` enables the shuffled-label null control.

#### Step 3: Generate the parent vs target comparison report

```bash
python scripts/reports/compare_parent_target_advantage.py \
  --parent-results outputs/runs/mimir_github_pythia_mink/results.json \
  --target-results outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full/results.json \
  --output-dir outputs/reports/target_membership_advantage_nnheui_pythia_1_4b_sft_full
```

Report: `outputs/reports/target_membership_advantage_nnheui_pythia_1_4b_sft_full/summary.md`

---

### Experiment 3 — Nonmember-vs-nonmember null control

Both pseudo-classes are drawn from the MIMIR GitHub **nonmember** pool. Labels
are artificial — neither class is a real member. The distinguisher should show
near-zero advantage here.

#### Step 1: Prepare the null-control dataset

```bash
python scripts/data/prepare_mimir_github_nonmember_control.py \
  --config github \
  --ngram-split ngram_13_0.2 \
  --num-train-per-class 170 \
  --num-test-per-class 200 \
  --seed 0 \
  --output-dir data/processed/mimir_github_nonmember_control_seed0
```

#### Step 2: Score the parent model on control data

```bash
python scripts/scoring/extract_logprob_scores.py \
  --model EleutherAI/pythia-1.4b \
  --train-file data/processed/mimir_github_nonmember_control_seed0/train.jsonl \
  --test-file  data/processed/mimir_github_nonmember_control_seed0/test.jsonl \
  --output-dir data/scores/mimir_github_nonmember_control_seed0_pythia_1_4b \
  --min-k-pcts 5,10,20,40 \
  --batch-size 4
```

#### Step 3: Run parent threshold experiment on control data

```bash
python scripts/experiments/run_mia_experiment.py \
  --train-scores data/scores/mimir_github_nonmember_control_seed0_pythia_1_4b/train_scores.jsonl \
  --test-scores  data/scores/mimir_github_nonmember_control_seed0_pythia_1_4b/test_scores.jsonl \
  --output-dir   outputs/runs/nonmember_control_seed0_pythia_1_4b \
  --primary-score min_k_20_logprob
```

#### Step 4: Score the target model on control data

```bash
python scripts/scoring/extract_logprob_scores.py \
  --model nnheui/pythia-1.4b-sft-full \
  --train-file data/processed/mimir_github_nonmember_control_seed0/train.jsonl \
  --test-file  data/processed/mimir_github_nonmember_control_seed0/test.jsonl \
  --output-dir data/scores/mimir_github_nonmember_control_seed0_nnheui_pythia_1_4b_sft_full \
  --min-k-pcts 5,10,20,40 \
  --batch-size 4
```

#### Step 5: Run target threshold experiment on control data

```bash
python scripts/experiments/run_mia_experiment.py \
  --train-scores data/scores/mimir_github_nonmember_control_seed0_nnheui_pythia_1_4b_sft_full/train_scores.jsonl \
  --test-scores  data/scores/mimir_github_nonmember_control_seed0_nnheui_pythia_1_4b_sft_full/test_scores.jsonl \
  --output-dir   outputs/runs/nonmember_control_seed0_nnheui_pythia_1_4b_sft_full \
  --primary-score min_k_20_logprob
```

#### Step 6: Generate the null-control report

```bash
python scripts/reports/report_nonmember_control.py \
  --parent-results outputs/runs/nonmember_control_seed0_pythia_1_4b/results.json \
  --target-results outputs/runs/nonmember_control_seed0_nnheui_pythia_1_4b_sft_full/results.json \
  --data-manifest  data/processed/mimir_github_nonmember_control_seed0/manifest.json \
  --output-dir outputs/reports/nonmember_vs_nonmember_control_parent_target
```

Report: `outputs/reports/nonmember_vs_nonmember_control_parent_target/summary.md`

---

## Script Reference

### `prepare_mimir_github.py`

| Argument | Default | Description |
|---|---|---|
| `--config` | `github` | MIMIR dataset config |
| `--split` | `ngram_13_0.2` | MIMIR n-gram dedup split |
| `--num-train-per-class` | 100 | Examples per class in the calibration split |
| `--num-test-per-class` | 100 | Examples per class in the evaluation split |
| `--max-words` | 32 | Truncate texts to this many words |
| `--min-words` | 8 | Drop texts shorter than this |
| `--seed` | 0 | RNG seed |
| `--output-dir` | — | Where to write `train.jsonl`, `test.jsonl`, `manifest.json` |
| `--token` | `$HF_TOKEN` | HuggingFace API token |

### `extract_logprob_scores.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | `EleutherAI/pythia-1.4b` | HuggingFace model ID |
| `--train-file` | — | Path to `train.jsonl` from data preparation |
| `--test-file` | — | Path to `test.jsonl` from data preparation |
| `--output-dir` | — | Where to write `train_scores.jsonl`, `test_scores.jsonl` |
| `--min-k-pcts` | `5,10,20,40` | MIN-K percentages to compute |
| `--batch-size` | 1 | Inference batch size |
| `--dtype` | `auto` | Model dtype (`bfloat16`, `float16`, `float32`) |

### `run_mia_experiment.py`

| Argument | Default | Description |
|---|---|---|
| `--train-scores` | — | Path to `train_scores.jsonl` |
| `--test-scores` | — | Path to `test_scores.jsonl` |
| `--output-dir` | — | Where to write `results.json` and the Markdown report |
| `--primary-score` | `min_k_20_logprob` | Score used for the headline result |
| `--criterion` | `balanced_accuracy` | Metric to maximize during threshold selection |
| `--parent-results` | — | `results.json` from the parent run (enables transfer diagnostic) |
| `--run-shuffled-control` | off | Run the shuffled-label null control |

---

## Key Concepts

**MIN-K% PROB** — the average log-probability of the k% lowest-probability
tokens in a sequence. Higher values indicate the model assigns higher
probability to the text (more member-like).

**Shard advantage** — TPR − FPR at the calibrated threshold. Range [−1, 1];
0 means no advantage over random guessing.

**Threshold transfer** — applying the parent model's calibrated threshold
directly to target model scores. A non-zero advantage under the transferred
threshold indicates that the two models' log-probability scales are correlated.

**Shuffled-label control** — randomly permuting the calibration-split labels
before threshold selection, then evaluating on the true test labels. Expected
advantage ≈ 0 under the null.

**Nonmember-vs-nonmember control** — constructing both pseudo-classes from the
MIMIR nonmember pool. Expected advantage ≈ 0. This validates that the main
experiment's signal is tied to actual membership.

---

## Dataset Access

The MIMIR dataset is gated. Request access at
[huggingface.co/datasets/iamgroot42/mimir](https://huggingface.co/datasets/iamgroot42/mimir)
and set `HF_TOKEN` before running any script.

**Important:** the `datasets` library cannot load MIMIR directly (the dataset
uses a legacy loading script). This codebase uses `hf_hub_download` to fetch
the raw JSONL files directly, which bypasses the issue.

---

## Output Structure

```
data/
  processed/
    mimir_github/                    # Main member/nonmember dataset
    mimir_github_nonmember_control_seed0/   # Null-control dataset
  scores/
    mimir_github_pythia_mink/        # Parent model scores
    mimir_github_nnheui_pythia_1_4b_sft_full/  # Target model scores
    mimir_github_nonmember_control_seed0_*/    # Control scores

outputs/
  runs/
    mimir_github_pythia_mink/        # Parent experiment results
    mimir_github_nnheui_pythia_1_4b_sft_full/  # Target experiment results
    nonmember_control_seed0_*/       # Control experiment results
  reports/
    target_membership_advantage_*/   # Parent vs target comparison
    nonmember_vs_nonmember_control_*/ # Null-control report
```
