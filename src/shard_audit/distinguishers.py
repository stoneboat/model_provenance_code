"""Threshold-based distinguisher for shard membership auditing.

Phase 3: implements threshold sweep on MIA calibration (train) scores,
selects optimal threshold, and evaluates on held-out (test) scores.

Score convention: higher value = more member-like (label=1).
This holds for mean_logprob and min_k_*_logprob.
For mean_loss, use negated values (or flip labels).

Metrics reported:
  accuracy              = (TP + TN) / N
  balanced_accuracy     = 0.5 * (TPR + TNR)
  auc                   = area under ROC curve (sklearn)
  tpr_at_1_fpr          = TPR when FPR ≤ 1% (interpolated)
  shard_advantage       = TPR - FPR at chosen threshold
                          (in [−1,1]; 0 = no advantage over random)
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Threshold sweep
# ------------------------------------------------------------------ #

def sweep_thresholds(
    labels: list,
    scores: list,
    n_thresholds: int = 1000,
) -> list:
    """Sweep scalar thresholds and return a list of (threshold, accuracy, tpr, fpr) tuples.

    Predicts label=1 if score >= threshold.
    Returns results sorted by threshold ascending.
    """
    min_s, max_s = min(scores), max(scores)
    step = (max_s - min_s) / n_thresholds if max_s > min_s else 1.0
    thresholds = [min_s + i * step for i in range(n_thresholds + 1)]

    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos

    results = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
        tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
        fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
        acc = (tp + tn) / n
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        results.append({
            "threshold": t,
            "accuracy": acc,
            "balanced_accuracy": 0.5 * (tpr + (1 - fpr)),
            "tpr": tpr,
            "fpr": fpr,
            "advantage": tpr - fpr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return results


def select_threshold(
    sweep_results: list,
    criterion: str = "balanced_accuracy",
) -> dict:
    """Return the threshold entry that maximises the given criterion."""
    return max(sweep_results, key=lambda r: r[criterion])


# ------------------------------------------------------------------ #
# Evaluation at a fixed threshold
# ------------------------------------------------------------------ #

def evaluate_at_threshold(
    labels: list,
    scores: list,
    threshold: float,
) -> dict:
    """Evaluate classification metrics at a fixed threshold.

    Predicts label=1 if score >= threshold.
    """
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    accuracy = (tp + tn) / n
    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    tnr = tn / n_neg if n_neg > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    advantage = tpr - fpr

    return {
        "threshold": threshold,
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "shard_advantage": advantage,
    }


def tpr_at_fpr(
    labels: list,
    scores: list,
    target_fpr: float = 0.01,
) -> Optional[float]:
    """Return TPR at the lowest threshold where FPR ≤ target_fpr.

    Uses sklearn's roc_curve for interpolation.
    Returns None if sklearn unavailable or labels are single-class.
    """
    try:
        from sklearn.metrics import roc_curve  # noqa: PLC0415
        if len(set(labels)) < 2:
            return None
        fprs, tprs, _ = roc_curve(labels, scores)
        # Find highest TPR achievable at FPR <= target_fpr
        eligible = [(f, t) for f, t in zip(fprs, tprs) if f <= target_fpr]
        if not eligible:
            return 0.0
        return max(t for _, t in eligible)
    except Exception:
        return None


def safe_auc(labels: list, scores: list) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        if len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Full distinguisher evaluation
# ------------------------------------------------------------------ #

def run_distinguisher(
    train_labels: list,
    train_scores: list,
    test_labels: list,
    test_scores: list,
    score_name: str,
    criterion: str = "balanced_accuracy",
    n_thresholds: int = 1000,
) -> dict:
    """Calibrate threshold on train, evaluate on test.

    Returns a result dict with all metrics for both splits.
    """
    # Calibration
    sweep = sweep_thresholds(train_labels, train_scores, n_thresholds)
    best = select_threshold(sweep, criterion=criterion)
    threshold = best["threshold"]

    # Train metrics at calibrated threshold
    train_metrics = evaluate_at_threshold(train_labels, train_scores, threshold)
    train_auc = safe_auc(train_labels, train_scores)
    train_tpr1fpr = tpr_at_fpr(train_labels, train_scores, 0.01)

    # Test metrics at calibrated threshold
    test_metrics = evaluate_at_threshold(test_labels, test_scores, threshold)
    test_auc = safe_auc(test_labels, test_scores)
    test_tpr1fpr = tpr_at_fpr(test_labels, test_scores, 0.01)

    logger.info(
        "[%s] threshold=%.4f | train acc=%.3f bal=%.3f adv=%.3f | "
        "test acc=%.3f bal=%.3f adv=%.3f auc=%.3f",
        score_name, threshold,
        train_metrics["accuracy"], train_metrics["balanced_accuracy"],
        train_metrics["shard_advantage"],
        test_metrics["accuracy"], test_metrics["balanced_accuracy"],
        test_metrics["shard_advantage"],
        test_auc or float("nan"),
    )

    return {
        "score_name": score_name,
        "calibration_criterion": criterion,
        "calibrated_threshold": threshold,
        "train": {**train_metrics, "auc": train_auc, "tpr_at_1_fpr": train_tpr1fpr},
        "test": {**test_metrics, "auc": test_auc, "tpr_at_1_fpr": test_tpr1fpr},
    }
