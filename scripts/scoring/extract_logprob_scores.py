"""Extract per-token log-probability scores from a causal LM for MIA.

Loads train.jsonl and test.jsonl produced by prepare_mimir_github.py,
scores each example with the parent model (default: EleutherAI/pythia-1.4b),
and writes JSONL score files plus debug examples.

Usage:
    python scripts/scoring/extract_logprob_scores.py \\
        --model EleutherAI/pythia-1.4b \\
        --train-file data/processed/mimir_github_phase2_tiny/train.jsonl \\
        --test-file  data/processed/mimir_github_phase2_tiny/test.jsonl \\
        --output-dir data/scores/mimir_github_pythia_mink_phase2_tiny \\
        --min-k-pcts 5,10,20,40 \\
        --batch-size 1 \\
        --debug-examples 6
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.logprobs import (
    get_device,
    load_model_and_tokenizer,
    extract_token_logprobs,
    build_debug_record,
)
from src.shard_audit.mia_scores import compute_all_scores
from src.shard_audit.metrics import score_diagnostics
from src.shard_audit.sanity_checks import check_score_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _token_length_histogram(lengths: list) -> dict:
    import collections
    bins = [0, 10, 20, 30, 40, 64, 128, 256, 512, 999999]
    counts = collections.Counter()
    for l in lengths:
        for lo, hi in zip(bins, bins[1:]):
            if lo <= l < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


# ------------------------------------------------------------------ #
# Core scoring loop
# ------------------------------------------------------------------ #

def score_records(
    records: list,
    model,
    tokenizer,
    device,
    k_pcts: list,
    batch_size: int,
    debug_n: int,
    model_name: str,
    max_length: int = 512,
) -> tuple:
    """Score a list of records and return (score_records, debug_records)."""
    score_results = []
    debug_results = []
    debug_count = 0

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        texts = [r["text"] for r in batch]

        # Extract per-token log-probs
        batch_lp = extract_token_logprobs(texts, model, tokenizer, device, max_length=max_length)

        for rec, lp in zip(batch, batch_lp):
            scores = compute_all_scores(rec["text"], lp, k_pcts=tuple(k_pcts))
            score_rec = {
                "id": rec["id"],
                "label": rec["label"],
                "phase_split": rec.get("phase_split", ""),
                "text_hash": rec["text_hash"],
                "model": model_name,
                "num_input_tokens": len(lp) + 1,  # +1 for the unscored first token
                "num_scored_tokens": len(lp),
                **{f"mean_logprob": round(scores["mean_logprob"], 6)},
                **{f"mean_loss": round(scores["mean_loss"], 6)},
                **{
                    f"min_k_{k}_logprob": round(scores[f"min_k_{k}_logprob"], 6)
                    for k in k_pcts
                },
                "zlib_norm_logprob": scores.get("zlib_norm_logprob"),
            }
            score_results.append(score_rec)

            if debug_count < debug_n:
                dbg = build_debug_record(
                    rec["text"], lp, tokenizer, rec["id"], rec["label"]
                )
                debug_results.append(dbg)
                debug_count += 1

        if (batch_start // batch_size + 1) % 10 == 0:
            logger.info(
                "Scored %d / %d examples...",
                min(batch_start + batch_size, len(records)),
                len(records),
            )

    return score_results, debug_results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-token log-probability scores from a causal LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--train-file", default="data/processed/mimir_github/train.jsonl")
    p.add_argument("--test-file", default="data/processed/mimir_github/test.jsonl")
    p.add_argument("--output-dir", default="data/scores/mimir_github_pythia_mink")
    p.add_argument("--min-k-pcts", default="5,10,20,40",
                   help="Comma-separated MIN-K percentages")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap total examples per split (for smoke tests)")
    p.add_argument("--debug-examples", type=int, default=6,
                   help="Number of debug records to write")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    k_pcts = [int(k) for k in args.min_k_pcts.split(",")]

    logger.info("=== Log-Probability Score Extraction ===")
    logger.info("Model:       %s", args.model)
    logger.info("Train file:  %s", args.train_file)
    logger.info("Test file:   %s", args.test_file)
    logger.info("Output dir:  %s", args.output_dir)
    logger.info("MIN-K pcts:  %s", k_pcts)
    logger.info("Batch size:  %d", args.batch_size)
    logger.info("Device:      %s", args.device)
    logger.info("Dtype:       %s", args.dtype)
    logger.info("Max examples: %s", args.max_examples)

    # Load data
    train_records = _load_jsonl(args.train_file)
    test_records = _load_jsonl(args.test_file)

    if args.max_examples:
        train_records = train_records[: args.max_examples]
        test_records = test_records[: args.max_examples]
        logger.info("Capped to %d train, %d test examples.", len(train_records), len(test_records))

    # Load model
    logger.info("\nLoading model %s...", args.model)
    device = get_device(args.device)
    logger.info("Device: %s", device)
    model, tokenizer = load_model_and_tokenizer(args.model, device, args.dtype)

    import torch
    actual_dtype = next(model.parameters()).dtype
    logger.info("Model dtype: %s", actual_dtype)

    # Score
    logger.info("\nScoring %d train examples...", len(train_records))
    train_scores, train_debug = score_records(
        train_records, model, tokenizer, device,
        k_pcts=k_pcts, batch_size=args.batch_size,
        debug_n=args.debug_examples, model_name=args.model,
        max_length=args.max_length,
    )

    logger.info("Scoring %d test examples...", len(test_records))
    test_scores, _ = score_records(
        test_records, model, tokenizer, device,
        k_pcts=k_pcts, batch_size=args.batch_size,
        debug_n=0, model_name=args.model,
        max_length=args.max_length,
    )

    # Sanity checks on scores
    train_sc_check = check_score_records(train_scores, k_pcts=tuple(k_pcts))
    test_sc_check = check_score_records(test_scores, k_pcts=tuple(k_pcts))
    if not train_sc_check["ok"]:
        logger.warning("Score issues in train: %s", train_sc_check["score_issues"][:5])
    if not test_sc_check["ok"]:
        logger.warning("Score issues in test: %s", test_sc_check["score_issues"][:5])

    # Diagnostics
    all_scores = train_scores + test_scores
    tok_lengths = [r["num_scored_tokens"] for r in all_scores]
    tok_hist = _token_length_histogram(tok_lengths)

    score_keys = ["mean_logprob", "mean_loss"] + [f"min_k_{k}_logprob" for k in k_pcts]
    train_diag = score_diagnostics(train_scores, score_keys=tuple(score_keys), split_name="train")
    test_diag = score_diagnostics(test_scores, score_keys=tuple(score_keys), split_name="test")

    logger.info("\n=== Score Diagnostics (Train) ===")
    for key, vals in train_diag.items():
        logger.info(
            "  %-25s  label0=%+.4f  label1=%+.4f  AUC=%.4f  dir=%s",
            key,
            vals.get("mean_label_0", float("nan")),
            vals.get("mean_label_1", float("nan")),
            vals.get("auc") or float("nan"),
            "OK" if vals.get("direction_ok") else "INVERTED",
        )

    logger.info("\n=== Score Diagnostics (Test) ===")
    for key, vals in test_diag.items():
        logger.info(
            "  %-25s  label0=%+.4f  label1=%+.4f  AUC=%.4f  dir=%s",
            key,
            vals.get("mean_label_0", float("nan")),
            vals.get("mean_label_1", float("nan")),
            vals.get("auc") or float("nan"),
            "OK" if vals.get("direction_ok") else "INVERTED",
        )

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_scores, os.path.join(args.output_dir, "train_scores.jsonl"))
    _write_jsonl(test_scores, os.path.join(args.output_dir, "test_scores.jsonl"))
    _write_jsonl(train_debug, os.path.join(args.output_dir, "debug_examples.jsonl"))

    manifest = {
        "model": args.model,
        "train_file": os.path.abspath(args.train_file),
        "test_file": os.path.abspath(args.test_file),
        "device": str(device),
        "dtype": str(actual_dtype),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "k_pcts": k_pcts,
        "n_train_scored": len(train_scores),
        "n_test_scored": len(test_scores),
        "token_length_histogram": tok_hist,
        "score_diagnostics_train": {
            k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                for kk, vv in v.items()}
            for k, v in train_diag.items()
        },
        "score_diagnostics_test": {
            k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                for kk, vv in v.items()}
            for k, v in test_diag.items()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\n=== DONE ===")
    logger.info("Output: %s", args.output_dir)
    logger.info("  train_scores.jsonl:   %d", len(train_scores))
    logger.info("  test_scores.jsonl:    %d", len(test_scores))
    logger.info("  debug_examples.jsonl: %d", len(train_debug))
    logger.info("  manifest.json:        written")


if __name__ == "__main__":
    main()
