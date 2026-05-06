"""Sanity checks for MIMIR GitHub / Pythia shard membership experiment.

Run before and after dataset preparation to catch common mistakes:
label imbalance, preprocessing inconsistency, exact-text overlaps, etc.
"""

import hashlib
import math
from typing import Optional


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_label_balance(records: list, name: str = "split") -> None:
    """Assert equal numbers of label-1 and label-0 examples."""
    n1 = sum(1 for r in records if r.get("label") == 1)
    n0 = sum(1 for r in records if r.get("label") == 0)
    assert n1 == n0, (
        f"[{name}] Label imbalance: {n1} member vs {n0} nonmember. "
        "Splits must be class-balanced."
    )


def check_no_text_overlap(
    records_a: list,
    records_b: list,
    name_a: str = "set_a",
    name_b: str = "set_b",
    allow_overlap: bool = False,
) -> int:
    """Check for exact text hash overlap between two record lists.

    Returns the number of overlapping hashes.
    Raises ValueError if allow_overlap is False and overlap > 0.
    """
    hashes_a = {r["text_hash"] for r in records_a}
    hashes_b = {r["text_hash"] for r in records_b}
    overlap = hashes_a & hashes_b
    if overlap and not allow_overlap:
        raise ValueError(
            f"Exact text overlap between {name_a} and {name_b}: "
            f"{len(overlap)} duplicate(s). This would leak information."
        )
    return len(overlap)


def check_word_counts(
    records: list,
    min_words: int = 8,
    max_words: int = 32,
    name: str = "records",
) -> None:
    """Assert all records satisfy word-count constraints after preprocessing."""
    for i, r in enumerate(records):
        text = r.get("text", "")
        words = len(text.split())
        assert min_words <= words <= max_words, (
            f"[{name}][{i}] Word count {words} outside [{min_words}, {max_words}]: "
            f"{text[:60]!r}"
        )


def check_required_fields(records: list, required: tuple = ("id", "text", "label", "text_hash")) -> None:
    """Assert every record has the required fields."""
    for i, r in enumerate(records):
        for field in required:
            assert field in r, f"Record[{i}] missing field '{field}': {list(r.keys())}"


def check_preprocessing_consistency(
    member_records: list,
    nonmember_records: list,
    min_words: int = 8,
    max_words: int = 32,
) -> dict:
    """Run all preprocessing sanity checks and return a summary dict."""
    issues = []

    for cls_name, records in (("member", member_records), ("nonmember", nonmember_records)):
        for r in records:
            text = r.get("text", "")
            wc = len(text.split())
            if wc < min_words:
                issues.append(f"{cls_name} id={r.get('id')} too short ({wc} words)")
            if wc > max_words:
                issues.append(f"{cls_name} id={r.get('id')} too long ({wc} words)")

    overlap_count = check_no_text_overlap(
        member_records, nonmember_records,
        name_a="member", name_b="nonmember",
        allow_overlap=True,  # report but don't abort; caller decides
    )

    return {
        "preprocessing_issues": issues,
        "member_nonmember_overlap_count": overlap_count,
        "ok": len(issues) == 0,
    }


def check_score_records(records: list, k_pcts: tuple = (5, 10, 20, 40)) -> dict:
    """Check score records for NaN, inf, and sign anomalies."""
    issues = []
    for r in records:
        for k in k_pcts:
            key = f"min_k_{k}_logprob"
            val = r.get(key)
            if val is None:
                continue
            if math.isnan(val) or math.isinf(val):
                issues.append(f"id={r.get('id')} {key}={val}")
            if val > 0:
                issues.append(f"id={r.get('id')} {key}={val:.4f} > 0 (logprob should be <=0)")
    return {"score_issues": issues, "ok": len(issues) == 0}
