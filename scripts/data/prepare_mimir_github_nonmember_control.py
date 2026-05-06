"""Prepare nonmember-vs-nonmember control data for shard-membership auditing.

Both pseudo-classes are drawn exclusively from the MIMIR GitHub nonmember/test
side. Labels are ARTIFICIAL — they do NOT indicate membership.

  label=1 / control_label='nonmember_a'  → pseudo-shard S0
  label=0 / control_label='nonmember_b'  → pseudo-shard S1

Both S0 and S1 are nonmembers w.r.t. the Pythia pretraining corpus.

Scientific purpose:
    The distinguisher should show near-zero advantage here.
    A positive result in the main experiment combined with near-zero result
    in this control supports that the main signal is tied to actual membership.

Usage:
    python scripts/data/prepare_mimir_github_nonmember_control.py \\
        --num-train-per-class 170 \\
        --num-test-per-class 200 \\
        --seed 0 \\
        --output-dir data/processed/mimir_github_nonmember_control_seed0
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.datasets import (
    DATASET_ID,
    load_mimir_github_nonmembers,
    _nonmember_filename,
)
from src.shard_audit.preprocessing import preprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Main experiment sizes for comparison in the report
MAIN_EXPERIMENT_SIZES = {
    "num_train_per_class": 500,
    "num_test_per_class": 200,
}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_count_histogram(texts: list) -> dict:
    bins = [0, 8, 16, 24, 32, 48, 64, 128, 999999]
    counts = Counter()
    for text in texts:
        wc = len(text.split())
        for lo, hi in zip(bins, bins[1:]):
            if lo <= wc < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


def _preprocess_and_dedup(texts: list, max_words: int, min_words: int) -> tuple:
    """Return (clean_texts, n_filtered, n_deduped)."""
    kept, n_filtered, n_deduped = [], 0, 0
    seen = set()
    for t in texts:
        result = preprocess_text(t, max_words=max_words, min_words=min_words)
        if result is None:
            n_filtered += 1
            continue
        h = _text_hash(result)
        if h in seen:
            n_deduped += 1
            continue
        seen.add(h)
        kept.append(result)
    return kept, n_filtered, n_deduped


def _determine_sizes(n_available: int, requested_train: int, requested_test: int) -> tuple:
    """Return (n_train, n_test) per class, with deterministic fallback."""
    per_class = requested_train + requested_test
    total_needed = per_class * 2
    if n_available >= total_needed:
        return requested_train, requested_test
    # Fallback 1: reduce train to 150, keep test
    fallback1_train = 150
    fallback1_total = (fallback1_train + requested_test) * 2
    if n_available >= fallback1_total:
        logger.warning(
            "Not enough examples for %d train+%d test per class (%d needed, %d available)."
            " Falling back to %d train + %d test per class.",
            requested_train, requested_test, total_needed, n_available,
            fallback1_train, requested_test,
        )
        return fallback1_train, requested_test
    # Fallback 2: 150 train + 150 test
    fallback2_train, fallback2_test = 150, 150
    fallback2_total = (fallback2_train + fallback2_test) * 2
    if n_available >= fallback2_total:
        logger.warning(
            "Falling back to %d train + %d test per class.", fallback2_train, fallback2_test
        )
        return fallback2_train, fallback2_test
    # Minimum viable: floor(n_available/4) each
    min_per = max(10, n_available // 4)
    if n_available >= min_per * 4:
        logger.warning(
            "Very small pool (%d). Using %d train + %d test per class.",
            n_available, min_per, min_per,
        )
        return min_per, min_per
    raise ValueError(
        f"Cannot construct a 2-class control: only {n_available} examples available."
    )


def _make_control_record(
    text: str,
    idx: int,
    label: int,
    control_label: str,
    phase_split: str,
) -> dict:
    return {
        "id": f"{control_label}-{idx:06d}",
        "text": text,
        "label": label,
        "control_label": control_label,
        "true_membership": "nonmember",
        "source": "mimir_github_nonmember_control",
        "split_origin": "mimir_test_nonmember",
        "phase_split": phase_split,
        "text_hash": _text_hash(text),
    }


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare MIMIR GitHub nonmember-vs-nonmember control dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--config", default="github")
    p.add_argument("--ngram-split", default="ngram_13_0.2",
                   dest="ngram_split")
    p.add_argument("--num-train-per-class", type=int, default=170)
    p.add_argument("--num-test-per-class", type=int, default=200)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir",
                   default="data/processed/mimir_github_nonmember_control_seed0")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=== MIMIR GitHub Nonmember-vs-Nonmember Control Data Preparation ===")
    logger.info("Dataset:       %s", args.dataset_id)
    logger.info("Config:        %s", args.config)
    logger.info("N-gram split:  %s", args.ngram_split)
    logger.info("Requested:     %d train + %d test per pseudo-class",
                args.num_train_per_class, args.num_test_per_class)
    logger.info("max_words:     %d", args.max_words)
    logger.info("min_words:     %d", args.min_words)
    logger.info("Seed:          %d", args.seed)
    logger.info("Output dir:    %s", args.output_dir)
    logger.info("")
    logger.info("LABEL SEMANTICS:")
    logger.info("  label=1 / control_label='nonmember_a' = pseudo-shard S0 (nonmember)")
    logger.info("  label=0 / control_label='nonmember_b' = pseudo-shard S1 (nonmember)")
    logger.info("  Both classes are nonmembers w.r.t. Pythia pretraining.")

    # ------------------------------------------------------------------ #
    # 1. Load nonmember texts
    # ------------------------------------------------------------------ #
    logger.info("\n[1/5] Loading MIMIR GitHub nonmember texts...")
    nonmember_texts_raw = load_mimir_github_nonmembers(
        config=args.config,
        split=args.ngram_split,
        token=args.token,
    )
    n_raw = len(nonmember_texts_raw)
    logger.info("Raw nonmember texts: %d", n_raw)
    wc_before = _word_count_histogram(nonmember_texts_raw)

    # ------------------------------------------------------------------ #
    # 2. Preprocess and deduplicate
    # ------------------------------------------------------------------ #
    logger.info("\n[2/5] Preprocessing (min=%d, max=%d words)...",
                args.min_words, args.max_words)
    clean_texts, n_filtered, n_deduped = _preprocess_and_dedup(
        nonmember_texts_raw, args.max_words, args.min_words,
    )
    n_available = len(clean_texts)
    logger.info("After preprocessing: %d kept, %d too-short, %d duplicates removed",
                n_available, n_filtered, n_deduped)
    wc_after = _word_count_histogram(clean_texts)

    # ------------------------------------------------------------------ #
    # 3. Determine final sizes with fallback
    # ------------------------------------------------------------------ #
    n_train, n_test = _determine_sizes(
        n_available, args.num_train_per_class, args.num_test_per_class
    )
    per_class = n_train + n_test
    total_used = per_class * 2
    logger.info("Using: %d train + %d test per pseudo-class (%d total used of %d available)",
                n_train, n_test, total_used, n_available)

    # ------------------------------------------------------------------ #
    # 4. Shuffle and split into two disjoint pseudo-shards
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Splitting into pseudo-shards S0 and S1...")
    rng = random.Random(args.seed)
    shuffled = list(clean_texts)
    rng.shuffle(shuffled)

    s0_pool = shuffled[:per_class]
    s1_pool = shuffled[per_class : per_class * 2]

    # Verify disjoint by hash
    s0_hashes = {_text_hash(t) for t in s0_pool}
    s1_hashes = {_text_hash(t) for t in s1_pool}
    overlap = s0_hashes & s1_hashes
    assert len(overlap) == 0, f"S0/S1 hash overlap: {len(overlap)} texts"
    logger.info("S0 and S1 are disjoint by hash (no overlap).")

    # Each pool: first n_train → train, next n_test → test
    s0_train_texts = s0_pool[:n_train]
    s0_test_texts  = s0_pool[n_train:]
    s1_train_texts = s1_pool[:n_train]
    s1_test_texts  = s1_pool[n_train:]

    # Verify no train/test cross-shard overlap
    all_train_hashes = {_text_hash(t) for t in s0_train_texts + s1_train_texts}
    all_test_hashes  = {_text_hash(t) for t in s0_test_texts  + s1_test_texts}
    tt_overlap = all_train_hashes & all_test_hashes
    assert len(tt_overlap) == 0, f"Train/test hash overlap: {len(tt_overlap)} texts"
    logger.info("Train and test splits are disjoint by hash.")

    # ------------------------------------------------------------------ #
    # 5. Build records
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Building records...")
    train_records = []
    for i, t in enumerate(s0_train_texts):
        train_records.append(_make_control_record(t, i, label=1, control_label="nonmember_a",
                                                   phase_split="train"))
    for i, t in enumerate(s1_train_texts):
        train_records.append(_make_control_record(t, i, label=0, control_label="nonmember_b",
                                                   phase_split="train"))

    test_records = []
    for i, t in enumerate(s0_test_texts):
        test_records.append(_make_control_record(t, i, label=1, control_label="nonmember_a",
                                                  phase_split="test"))
    for i, t in enumerate(s1_test_texts):
        test_records.append(_make_control_record(t, i, label=0, control_label="nonmember_b",
                                                  phase_split="test"))

    rng.shuffle(train_records)
    rng.shuffle(test_records)

    # Final sanity checks
    n_train_a = sum(1 for r in train_records if r["label"] == 1)
    n_train_b = sum(1 for r in train_records if r["label"] == 0)
    n_test_a  = sum(1 for r in test_records  if r["label"] == 1)
    n_test_b  = sum(1 for r in test_records  if r["label"] == 0)
    assert n_train_a == n_train_b == n_train, f"Train imbalance: {n_train_a} vs {n_train_b}"
    assert n_test_a  == n_test_b  == n_test,  f"Test imbalance:  {n_test_a} vs {n_test_b}"

    all_hashes_train = {r["text_hash"] for r in train_records}
    all_hashes_test  = {r["text_hash"] for r in test_records}
    assert len(all_hashes_train & all_hashes_test) == 0, "Final train/test hash overlap"
    logger.info("All sanity checks passed.")

    # ------------------------------------------------------------------ #
    # 6. Write outputs
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Writing outputs to %s...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records,  os.path.join(args.output_dir, "test.jsonl"))

    # Finite-sample null bound
    null_bound_95 = math.sqrt(math.log(40) / n_test) if n_test > 0 else None

    manifest = {
        "experiment_type": "nonmember_vs_nonmember_control",
        "dataset_id": args.dataset_id,
        "config": args.config,
        "ngram_split": args.ngram_split,
        "source_file": _nonmember_filename(args.config, args.ngram_split),
        "label_semantics": {
            "label_1": "nonmember_a (pseudo-shard S0, drawn from MIMIR nonmember/test side)",
            "label_0": "nonmember_b (pseudo-shard S1, drawn from MIMIR nonmember/test side)",
            "true_membership_both_classes": "nonmember",
            "warning": (
                "These are artificial control labels. "
                "They do NOT indicate member vs nonmember. "
                "Both classes are nonmembers with respect to the Pythia pretraining membership label."
            ),
        },
        "preprocessing": {
            "max_words": args.max_words,
            "min_words": args.min_words,
        },
        "seed": args.seed,
        "n_raw": n_raw,
        "n_after_preprocessing": n_available,
        "n_filtered_too_short": n_filtered,
        "n_filtered_duplicate": n_deduped,
        "n_train_per_class": n_train,
        "n_test_per_class": n_test,
        "n_train_total": len(train_records),
        "n_test_total": len(test_records),
        "s0_s1_hash_overlap": 0,
        "train_test_hash_overlap": 0,
        "null_bound_alpha05": round(null_bound_95, 4) if null_bound_95 else None,
        "main_experiment_comparison": {
            "num_train_per_class": MAIN_EXPERIMENT_SIZES["num_train_per_class"],
            "num_test_per_class": MAIN_EXPERIMENT_SIZES["num_test_per_class"],
            "parent_min_k_20_advantage": 0.380,
            "target_min_k_20_advantage": 0.375,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(args.output_dir),
    }

    diagnostics = {
        "word_count_histogram": {
            "before_truncation": wc_before,
            "after_truncation": wc_after,
        },
        "filtering": {
            "n_raw": n_raw,
            "n_filtered_too_short_or_invalid": n_filtered,
            "n_filtered_intra_class_duplicate": n_deduped,
            "n_available_after_preprocessing": n_available,
        },
        "split_sizes": {
            "n_train_per_class": n_train,
            "n_test_per_class": n_test,
            "n_train_total": len(train_records),
            "n_test_total": len(test_records),
            "total_used": total_used,
            "total_unused": n_available - total_used,
        },
    }

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("manifest.json written.")
    logger.info("diagnostics.json written.")

    print("\n=== DONE ===")
    print(f"  Output: {args.output_dir}")
    print(f"  train.jsonl:  {len(train_records)} records "
          f"({n_train} nonmember_a + {n_train} nonmember_b)")
    print(f"  test.jsonl:   {len(test_records)} records "
          f"({n_test} nonmember_a + {n_test} nonmember_b)")
    print(f"  Null bound (alpha=0.05, m={n_test}): {null_bound_95:.4f}" if null_bound_95 else "")
    print("  LABEL WARNING: both classes are nonmembers. Labels are artificial.")


if __name__ == "__main__":
    main()
