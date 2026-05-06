"""Prepare MIMIR GitHub member/nonmember dataset for shard-membership auditing.

Dataset: iamgroot42/mimir  (config=github, split=ngram_13_0.2 by default)

MIMIR file layout (from mimir.py dataset script):
  cache_100_200_1000_512/train/<config>_<split>.jsonl  → MEMBER texts
  cache_100_200_1000_512/test/<config>_<split>.jsonl   → NONMEMBER texts

  "train" = examples the model was trained on (members, label=1)
  "test"  = examples the model was NOT trained on (nonmembers, label=0)
  These are NOT an MIA train/test split — that is constructed downstream.

Output directory structure:
    <output_dir>/train.jsonl      ← MIA calibration split (member + nonmember)
    <output_dir>/test.jsonl       ← MIA evaluation split (member + nonmember)
    <output_dir>/manifest.json
    <output_dir>/diagnostics.json

Usage:
    python scripts/data/prepare_mimir_github.py --list-configs
    python scripts/data/prepare_mimir_github.py --inspect --config github
    python scripts/data/prepare_mimir_github.py \\
        --num-train-per-class 20 --num-test-per-class 20 \\
        --output-dir data/processed/mimir_github_phase2_tiny
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.datasets import (
    DATASET_ID,
    KNOWN_CONFIGS,
    KNOWN_SPLITS,
    list_configs,
    list_splits,
    load_mimir_github,
    texts_to_records,
    _member_filename,
    _nonmember_filename,
)
from src.shard_audit.preprocessing import preprocess_text
from src.shard_audit.splitting import stratified_train_test_split
from src.shard_audit.sanity_checks import (
    check_label_balance,
    check_word_counts,
    check_required_fields,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _word_count_histogram(texts: list, bins: list = None) -> dict:
    if bins is None:
        bins = [0, 8, 16, 24, 32, 48, 64, 128, 256, 999999]
    counts = Counter()
    for text in texts:
        wc = len(text.split())
        for lo, hi in zip(bins, bins[1:]):
            if lo <= wc < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


def _exact_overlap(hashes_a: set, hashes_b: set) -> int:
    return len(hashes_a & hashes_b)


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
        description="Prepare MIMIR GitHub member/nonmember dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--config", default="github",
                   help="Dataset config/subset (e.g. 'github')")
    p.add_argument("--split", default="ngram_13_0.2",
                   help="N-gram split name (e.g. 'ngram_13_0.2')")
    p.add_argument("--num-train-per-class", type=int, default=100)
    p.add_argument("--num-test-per-class", type=int, default=100)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/processed/mimir_github")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace API token (default: $HF_TOKEN)")
    p.add_argument("--list-configs", action="store_true",
                   help="Print available configs and exit")
    p.add_argument("--inspect", action="store_true",
                   help="Print dataset schema info and exit without processing")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Allow exact text overlap between member and nonmember")
    return p.parse_args()


def cmd_list_configs(token):
    configs = list_configs(token=token)
    print(f"\nDataset: {DATASET_ID}")
    print("Available configs:")
    for c in configs:
        print(f"  {c}")
    print("\nNote: use --config <name> to select one.")


def cmd_inspect(config: str, split: str, token):
    print(f"\nDataset:         {DATASET_ID}")
    print(f"Selected config: {config}")
    print(f"Selected split:  {split}")

    print("\nAvailable configs:")
    configs = list_configs(token=token)
    for c in configs:
        marker = " <-- selected" if c == config else ""
        print(f"  {c}{marker}")

    print("\nAvailable n-gram splits (from known schema):")
    for s in KNOWN_SPLITS:
        marker = " <-- selected" if s == split else ""
        print(f"  {s}{marker}")

    print("\nSchema (columns per example):")
    print("  member              str  — member text (in Pythia training corpus)")
    print("  nonmember           str  — nonmember text (not in training corpus)")
    print("  member_neighbors    List[str]  — BERT paraphrases (not used here)")
    print("  nonmember_neighbors List[str]  — BERT paraphrases (not used here)")

    print("\nFile layout (from mimir.py):")
    print(f"  MEMBER    file: {_member_filename(config, split)}")
    print(f"  NONMEMBER file: {_nonmember_filename(config, split)}")

    print("\nLabel assignment:")
    print("  member column    → label=1")
    print("  nonmember column → label=0")

    print("\nNote: 'train'/'test' in MIMIR paths refer to IN/OUT of model training corpus,")
    print("  NOT to MIA train/test splits. The MIA split is created downstream.")


def main():
    args = parse_args()

    if args.list_configs:
        cmd_list_configs(token=args.token)
        sys.exit(0)

    if args.inspect:
        cmd_inspect(config=args.config, split=args.split, token=args.token)
        sys.exit(0)

    logger.info("=== MIMIR GitHub Dataset Preparation ===")
    logger.info("Dataset:       %s", args.dataset_id)
    logger.info("Config:        %s", args.config)
    logger.info("Split:         %s", args.split)
    logger.info("Train/class:   %d", args.num_train_per_class)
    logger.info("Test/class:    %d", args.num_test_per_class)
    logger.info("max_words:     %d", args.max_words)
    logger.info("min_words:     %d", args.min_words)
    logger.info("Seed:          %d", args.seed)
    logger.info("Output dir:    %s", args.output_dir)

    # ------------------------------------------------------------------ #
    # 1. Load raw texts
    # ------------------------------------------------------------------ #
    logger.info("\n[1/5] Loading MIMIR GitHub texts via hub download...")
    logger.info("  Member    file: %s", _member_filename(args.config, args.split))
    logger.info("  Nonmember file: %s", _nonmember_filename(args.config, args.split))

    member_texts_raw, nonmember_texts_raw = load_mimir_github(
        config=args.config,
        split=args.split,
        token=args.token,
    )
    logger.info(
        "Raw loaded: %d members, %d nonmembers",
        len(member_texts_raw), len(nonmember_texts_raw),
    )

    n_member_raw = len(member_texts_raw)
    n_nonmember_raw = len(nonmember_texts_raw)

    wc_member_before = _word_count_histogram(member_texts_raw)
    wc_nonmember_before = _word_count_histogram(nonmember_texts_raw)

    # ------------------------------------------------------------------ #
    # 2. Preprocess + deduplicate within each class
    # ------------------------------------------------------------------ #
    logger.info("\n[2/5] Preprocessing (min=%d, max=%d words)...", args.min_words, args.max_words)

    def _preprocess_list(texts):
        """Preprocess and deduplicate by truncated text hash (order-preserving)."""
        import hashlib
        kept, filtered, deduped = [], 0, 0
        seen_hashes = set()
        for t in texts:
            result = preprocess_text(t, max_words=args.max_words, min_words=args.min_words)
            if result is None:
                filtered += 1
                continue
            h = hashlib.sha256(result.encode()).hexdigest()
            if h in seen_hashes:
                deduped += 1
                continue
            seen_hashes.add(h)
            kept.append(result)
        return kept, filtered, deduped

    member_texts_clean, n_member_filtered, n_member_deduped = _preprocess_list(member_texts_raw)
    nonmember_texts_clean, n_nonmember_filtered, n_nonmember_deduped = _preprocess_list(nonmember_texts_raw)
    logger.info(
        "Members:    %d kept, %d too-short, %d intra-class duplicates removed",
        len(member_texts_clean), n_member_filtered, n_member_deduped,
    )
    logger.info(
        "Nonmembers: %d kept, %d too-short, %d intra-class duplicates removed",
        len(nonmember_texts_clean), n_nonmember_filtered, n_nonmember_deduped,
    )

    logger.info(
        "After preprocessing: %d members (filtered %d), %d nonmembers (filtered %d)",
        len(member_texts_clean), n_member_filtered,
        len(nonmember_texts_clean), n_nonmember_filtered,
    )

    wc_member_after = _word_count_histogram(member_texts_clean)
    wc_nonmember_after = _word_count_histogram(nonmember_texts_clean)

    # ------------------------------------------------------------------ #
    # 3. Build records and check overlap
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Building records and checking overlaps...")

    member_records = texts_to_records(
        member_texts_clean, label=1,
        source="mimir_github", split_origin="member",
        id_prefix="member-",
    )
    nonmember_records = texts_to_records(
        nonmember_texts_clean, label=0,
        source="mimir_github", split_origin="nonmember",
        id_prefix="nonmember-",
    )

    member_hashes = {r["text_hash"] for r in member_records}
    nonmember_hashes = {r["text_hash"] for r in nonmember_records}
    overlap_count = _exact_overlap(member_hashes, nonmember_hashes)

    if overlap_count > 0:
        msg = (
            f"Exact text overlap: {overlap_count} texts appear in both "
            f"member and nonmember sets."
        )
        if args.allow_overlap:
            logger.warning(msg + " Continuing because --allow-overlap is set.")
        else:
            logger.error(msg + " Use --allow-overlap to proceed anyway.")
            sys.exit(1)
    else:
        logger.info("No exact text overlap between member and nonmember sets.")

    # ------------------------------------------------------------------ #
    # 4. Split
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Creating MIA train/test splits...")
    total_needed = args.num_train_per_class + args.num_test_per_class
    for cls_name, records in (("member", member_records), ("nonmember", nonmember_records)):
        if len(records) < total_needed:
            logger.error(
                "Not enough %s records: need %d, have %d. "
                "Reduce --num-train-per-class or --num-test-per-class.",
                cls_name, total_needed, len(records),
            )
            sys.exit(1)

    train_records, test_records = stratified_train_test_split(
        member_records=member_records,
        nonmember_records=nonmember_records,
        num_train_per_class=args.num_train_per_class,
        num_test_per_class=args.num_test_per_class,
        seed=args.seed,
    )

    n_train_member = sum(1 for r in train_records if r["label"] == 1)
    n_train_nonmember = sum(1 for r in train_records if r["label"] == 0)
    n_test_member = sum(1 for r in test_records if r["label"] == 1)
    n_test_nonmember = sum(1 for r in test_records if r["label"] == 0)

    logger.info("MIA train: %d total (%d member, %d nonmember)",
                len(train_records), n_train_member, n_train_nonmember)
    logger.info("MIA test:  %d total (%d member, %d nonmember)",
                len(test_records), n_test_member, n_test_nonmember)

    train_hashes = {r["text_hash"] for r in train_records}
    test_hashes = {r["text_hash"] for r in test_records}
    tt_overlap = len(train_hashes & test_hashes)
    assert tt_overlap == 0, f"MIA train/test hash overlap: {tt_overlap}"
    logger.info("No MIA train/test hash overlap.")

    check_label_balance(train_records, name="train")
    check_label_balance(test_records, name="test")
    check_required_fields(train_records)
    check_required_fields(test_records)
    check_word_counts(train_records, min_words=args.min_words, max_words=args.max_words)
    check_word_counts(test_records, min_words=args.min_words, max_words=args.max_words)
    logger.info("All sanity checks passed.")

    # ------------------------------------------------------------------ #
    # 5. Write outputs
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Writing outputs to %s...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records, os.path.join(args.output_dir, "test.jsonl"))

    manifest = {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "member_source_file": _member_filename(args.config, args.split),
        "nonmember_source_file": _nonmember_filename(args.config, args.split),
        "schema_columns": ["member", "nonmember", "member_neighbors", "nonmember_neighbors"],
        "member_column": "member (train/ in MIMIR)",
        "nonmember_column": "nonmember (test/ in MIMIR)",
        "member_label": 1,
        "nonmember_label": 0,
        "max_words": args.max_words,
        "min_words": args.min_words,
        "num_train_per_class": args.num_train_per_class,
        "num_test_per_class": args.num_test_per_class,
        "seed": args.seed,
        "n_member_raw": n_member_raw,
        "n_nonmember_raw": n_nonmember_raw,
        "n_member_after_preprocessing": len(member_texts_clean),
        "n_nonmember_after_preprocessing": len(nonmember_texts_clean),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_train_member": n_train_member,
        "n_train_nonmember": n_train_nonmember,
        "n_test_member": n_test_member,
        "n_test_nonmember": n_test_nonmember,
        "member_nonmember_overlap": overlap_count,
        "mia_train_test_overlap": tt_overlap,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(args.output_dir),
    }

    diagnostics = {
        "word_count_histogram": {
            "member_before_truncation": wc_member_before,
            "nonmember_before_truncation": wc_nonmember_before,
            "member_after_truncation": wc_member_after,
            "nonmember_after_truncation": wc_nonmember_after,
        },
        "filtering": {
            "member_filtered_too_short_or_invalid": n_member_filtered,
            "member_filtered_intra_class_duplicate": n_member_deduped,
            "nonmember_filtered_too_short_or_invalid": n_nonmember_filtered,
            "nonmember_filtered_intra_class_duplicate": n_nonmember_deduped,
        },
        "overlap": {
            "member_nonmember_exact_text": overlap_count,
            "mia_train_test_exact_text": tt_overlap,
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
    print(f"  train.jsonl:   {len(train_records)} records")
    print(f"  test.jsonl:    {len(test_records)} records")
    print(f"  Overlap (member/nonmember): {overlap_count}")
    print(f"  Overlap (MIA train/test):   {tt_overlap}")


if __name__ == "__main__":
    main()
