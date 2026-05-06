"""Train/test splitting utilities for shard membership auditing.

Produces class-balanced, deterministic splits with no text-hash overlap
between train and test sets.
"""

import hashlib
import random
from typing import Optional


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stratified_train_test_split(
    member_records: list,
    nonmember_records: list,
    num_train_per_class: int,
    num_test_per_class: int,
    seed: int = 0,
) -> tuple:
    """Split member and nonmember records into balanced train/test sets.

    Args:
        member_records: dicts with at least 'text' and 'label'==1
        nonmember_records: dicts with at least 'text' and 'label'==0
        num_train_per_class: examples per class in train
        num_test_per_class: examples per class in test
        seed: RNG seed for determinism

    Returns:
        (train_records, test_records) — each a shuffled mix of member+nonmember

    Raises:
        ValueError if there are not enough examples or hash overlap exists.
    """
    total_needed = num_train_per_class + num_test_per_class
    for name, records in (("member", member_records), ("nonmember", nonmember_records)):
        if len(records) < total_needed:
            raise ValueError(
                f"Not enough {name} records: need {total_needed}, got {len(records)}"
            )

    rng = random.Random(seed)

    def _select(records, n_train, n_test):
        shuffled = list(records)
        rng.shuffle(shuffled)
        train = shuffled[:n_train]
        test = shuffled[n_train : n_train + n_test]
        return train, test

    m_train, m_test = _select(member_records, num_train_per_class, num_test_per_class)
    nm_train, nm_test = _select(nonmember_records, num_train_per_class, num_test_per_class)

    # Verify no hash overlap between train and test
    train_hashes = {r["text_hash"] for r in m_train + nm_train}
    test_hashes = {r["text_hash"] for r in m_test + nm_test}
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"Hash overlap between train and test: {len(overlap)} examples. "
            "This indicates near-duplicate or identical texts across the MIMIR splits."
        )

    def _tag(records, phase):
        tagged = []
        for r in records:
            tagged.append({**r, "phase_split": phase})
        return tagged

    train_all = _tag(m_train, "train") + _tag(nm_train, "train")
    test_all = _tag(m_test, "test") + _tag(nm_test, "test")

    rng.shuffle(train_all)
    rng.shuffle(test_all)

    return train_all, test_all
