"""Text preprocessing for shard-membership auditing.

Applies paper-matched preprocessing: whitespace normalization, word-count
filtering (min_words=8), and prefix truncation (max_words=32).
Both member and nonmember examples receive identical treatment.
"""

import re


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Count whitespace-delimited words after normalization."""
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len(normalized.split())


def truncate_words(text: str, max_words: int = 32) -> str:
    """Return the first max_words whitespace-delimited words of normalized text."""
    normalized = normalize_text(text)
    words = normalized.split()
    return " ".join(words[:max_words])


def is_valid_text(text: str, min_words: int = 8) -> bool:
    """Return True iff the normalized text has at least min_words words."""
    return count_words(text) >= min_words


def preprocess_text(
    text: str,
    max_words: int = 32,
    min_words: int = 8,
) -> "str | None":
    """Normalize, filter by min_words, then truncate to max_words.

    Returns None if the text is too short after normalization.
    """
    if not isinstance(text, str):
        return None
    if not is_valid_text(text, min_words=min_words):
        return None
    return truncate_words(text, max_words=max_words)
