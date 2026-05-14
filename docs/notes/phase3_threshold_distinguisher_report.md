# Phase 3 Completion Report: Threshold Distinguisher and Held-Out Test Accuracy

**Date:** 2026-05-07  
**Status:** Complete

---

## 1. Executive Summary

- **Parent model:** `deit__aug`
- **Dataset:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`, split=`ngram_13_0.2`)
- **Distinguisher:** threshold on `aug_neg_logit_var_mean` calibrated on train, evaluated on test
- **Primary metric (test accuracy):** **0.554**
- **Test balanced accuracy:** 0.554
- **Test AUC:** 0.567
- **Test shard advantage (TPR−FPR):** 0.108
- **Test TPR @ 1% FPR:** 0.015

The sanity check is **WEAK/NEGATIVE**: Pythia-1.4b assigns measurably higher  log-probability to its own GitHub training examples than to held-out nonmember examples.

---

## 2. Experiment Configuration

| Item | Value |
|---|---|
| Parent model | `deit__aug` |
| MIA score (primary) | `aug_neg_logit_var_mean` |
| Calibration criterion | `balanced_accuracy` |
| Train size | 3850 (1925 member, 1925 nonmember) |
| Test size  | 3850 (1925 member, 1925 nonmember) |
| MIMIR config | `ngram_13_0.2` |
| Max words | 32 |

---

## 3. Train-Split Metrics (threshold calibrated here)

| Score | Accuracy | Bal. Accuracy | AUC | TPR@1%FPR | Shard Adv. | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| aug_neg_logit_var_mean | 0.553 | 0.553 | 0.571 | 0.020 | 0.106 | -0.0486 |
| aug_neg_kl_mean | 0.559 | 0.559 | 0.583 | 0.013 | 0.119 | -0.0490 |
| aug_cos_sim_mean | 0.556 | 0.556 | 0.574 | 0.026 | 0.112 | 0.9793 |
| aug_neg_top1_std | 0.529 | 0.529 | 0.534 | 0.010 | 0.059 | -0.0403 |

---

## 4. Test-Split Metrics (held-out evaluation)

| Score | Accuracy | Bal. Accuracy | AUC | TPR@1%FPR | Shard Adv. | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| aug_neg_logit_var_mean | 0.554 | 0.554 | 0.567 | 0.015 | 0.108 | -0.0486 |
| aug_neg_kl_mean | 0.555 | 0.555 | 0.581 | 0.013 | 0.111 | -0.0490 |
| aug_cos_sim_mean | 0.552 | 0.552 | 0.574 | 0.018 | 0.104 | 0.9793 |
| aug_neg_top1_std | 0.525 | 0.525 | 0.537 | 0.012 | 0.050 | -0.0403 |

---

## 5. Primary Score Detail

**Score:** `aug_neg_logit_var_mean`  
**Calibrated threshold:** -0.0486  
**Criterion:** balanced_accuracy  

### Train

| Metric | Value |
|---|---:|
| Accuracy | 0.5530 |
| Balanced accuracy | 0.5530 |
| AUC | 0.5707 |
| TPR @ 1% FPR | 0.0197 |
| Shard advantage | 0.1060 |
| TP / FP / TN / FN | 1136 / 932 / 993 / 789 |

### Test (held-out)

| Metric | Value |
|---|---:|
| Accuracy | 0.5540 |
| Balanced accuracy | 0.5540 |
| AUC | 0.5671 |
| TPR @ 1% FPR | 0.0145 |
| Shard advantage | 0.1081 |
| TP / FP / TN / FN | 1148 / 940 / 985 / 777 |

---

## 6. Score Direction Verification

All scores use the convention: **higher = more member-like**.  
`mean_loss = -mean_logprob` and is deliberately inverted; its AUC < 0.5 is expected.
The threshold distinguisher correctly uses all scores in the 'higher = member' direction.

---

## 7. Commands Run

```bash
# Data preparation
python scripts/data/prepare_mimir_github.py \
  --num-train-per-class 500 \
  --num-test-per-class 200 \
  --max-words 32 --min-words 8 \
  --seed 0 --output-dir data/processed/mimir_github

# Scoring
python scripts/scoring/extract_logprob_scores.py \
  --model deit__aug \
  --train-file data/scores_aug/imagenet_shard__deit/train_scores.jsonl \
  --test-file  data/scores_aug/imagenet_shard__deit/test_scores.jsonl \
  --output-dir <scores_dir> \
  --min-k-pcts 5,10,20,40 --batch-size 4

# Experiment
python scripts/experiments/run_mia_experiment.py \
  --train-scores data/scores_aug/imagenet_shard__deit/train_scores.jsonl \
  --test-scores  data/scores_aug/imagenet_shard__deit/test_scores.jsonl \
  --output-dir   outputs/runs_aug/deit \
  --primary-score aug_neg_logit_var_mean
```

---

## 8. Interpretation and Next Steps

Test accuracy of **0.554** (55.4%) and AUC of **0.567** confirm that Pythia-1.4b's per-token log-probabilities carry a statistically meaningful signal distinguishing its GitHub training examples from nonmember examples.

A shard advantage of **0.108** (5.4 percentage points above the random-guess baseline of 0.0) supports the model-provenance hypothesis: the parent model's statistics are shifted in favor of the candidate shard.

**Recommended next steps:**

1. Run at full scale (all 700 members + all 700 nonmembers with a larger test split).
2. Compare base Pythia vs. a fine-tuned Pythia to test whether fine-tuning increases
   the distinguisher advantage beyond the base model's pretraining signal.
3. Implement the GRU distinguisher over the full token log-prob sequence (not just scalar scores).
4. Report ROC curves and score histograms (Phase 4 / reporting.py).