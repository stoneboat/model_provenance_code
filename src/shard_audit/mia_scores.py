"""Membership-inference attack (MIA) score computation.

Implements MIN-K% PROB and related scalar scores derived from per-token
log-probabilities.

Sign convention (consistent throughout):
  Higher mean_logprob  → more member-like
  Higher min_k_logprob → more member-like
  Lower  mean_loss     → more member-like
"""

import math
import zlib
from typing import Optional


def mean_logprob(token_logprobs: list) -> float:
    """Average log-probability across scored tokens."""
    if not token_logprobs:
        return float("nan")
    return sum(token_logprobs) / len(token_logprobs)


def mean_loss(token_logprobs: list) -> float:
    """Negative mean log-probability (cross-entropy loss estimate)."""
    return -mean_logprob(token_logprobs)


def min_k_logprob(token_logprobs: list, k_pct: float) -> float:
    """Average log-probability of the lowest-probability k% tokens.

    Higher returned value means more member-like.

    The k lowest log-probabilities are the hardest-to-predict tokens.
    Members tend to have higher average min-k scores because the model
    has seen them and assigns higher probability even to their rare tokens.

    Args:
        token_logprobs: per-token log-probabilities (all values <= 0)
        k_pct: percentage in (0, 100]

    Returns:
        Mean of the k_pct% lowest log-probabilities, or nan if empty.
    """
    if not token_logprobs:
        return float("nan")
    k_count = max(1, math.ceil(len(token_logprobs) * k_pct / 100.0))
    lowest = sorted(token_logprobs)[:k_count]
    return sum(lowest) / len(lowest)


def zlib_normalized_logprob(text: str, token_logprobs: list) -> Optional[float]:
    """Log-prob normalized by zlib compression length (bits).

    Returns None if zlib is unavailable or text is empty.
    Higher value is more member-like.
    """
    if not text or not token_logprobs:
        return None
    try:
        compressed_len = len(zlib.compress(text.encode("utf-8")))
        if compressed_len == 0:
            return None
        total_logprob = sum(token_logprobs)
        return total_logprob / compressed_len
    except Exception:
        return None


def compute_all_scores(
    text: str,
    token_logprobs: list,
    k_pcts: tuple = (5, 10, 20, 40),
) -> dict:
    """Compute all scalar MIA scores for a single example.

    Returns a dict with keys:
        mean_logprob, mean_loss, min_k_{k}_logprob (for each k), zlib_norm_logprob
    """
    result = {
        "mean_logprob": mean_logprob(token_logprobs),
        "mean_loss": mean_loss(token_logprobs),
    }
    for k in k_pcts:
        result[f"min_k_{k}_logprob"] = min_k_logprob(token_logprobs, k)
    result["zlib_norm_logprob"] = zlib_normalized_logprob(text, token_logprobs)
    return result
