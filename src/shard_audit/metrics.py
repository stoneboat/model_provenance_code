"""Score diagnostics for shard membership auditing (Phase 2: no threshold tuning).

Computes raw class-means and continuous AUC for each MIA score.
Threshold selection and final held-out accuracy are Phase 3.
"""

from typing import Optional


def safe_auc(labels: list, scores: list) -> Optional[float]:
    """Compute ROC-AUC safely; returns None if sklearn is unavailable.

    Higher score = more member-like (label==1).
    If AUC < 0.5, the score direction may be inverted — caller should check.
    """
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        if len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def class_means(labels: list, scores: list) -> dict:
    """Return mean score per class label.

    Returns {'mean_label_0': float, 'mean_label_1': float, 'direction_ok': bool}
    direction_ok is True iff mean for label 1 > mean for label 0 (member-like higher).
    """
    pairs_0 = [s for l, s in zip(labels, scores) if l == 0]
    pairs_1 = [s for l, s in zip(labels, scores) if l == 1]

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    m0 = _mean(pairs_0)
    m1 = _mean(pairs_1)

    import math
    direction_ok = (not math.isnan(m0)) and (not math.isnan(m1)) and (m1 > m0)

    return {"mean_label_0": m0, "mean_label_1": m1, "direction_ok": direction_ok}


def score_diagnostics(
    records: list,
    score_keys: tuple = ("mean_logprob", "min_k_20_logprob", "mean_loss"),
    split_name: str = "",
) -> dict:
    """Compute class means and AUC for each score key on a list of score records.

    Args:
        records: list of dicts with 'label' (int) and score keys (float)
        score_keys: which score columns to evaluate
        split_name: 'train' or 'test' for display

    Returns:
        dict mapping score_key -> {class_means, auc, flipped_auc_if_low}
    """
    results = {}
    labels = [r["label"] for r in records]
    for key in score_keys:
        scores = [r.get(key, float("nan")) for r in records]
        cm = class_means(labels, scores)
        auc = safe_auc(labels, scores)
        entry = {**cm, "auc": auc}
        if auc is not None and auc < 0.5:
            flipped = safe_auc([1 - l for l in labels], scores)
            entry["flipped_auc"] = flipped
            entry["note"] = "AUC < 0.5; score direction may be inverted"
        results[key] = entry
    return results
