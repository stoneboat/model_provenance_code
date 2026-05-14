"""Extract per-image membership scores from a ViT for the shard audit.

Reads train.jsonl and test.jsonl produced by `prepare_imagenet_shard.py`,
loads the target model T, and scores each image under
    score(x, y) = log P_head( T_encoder(x) )[y]
plus auxiliary diagnostics. Writes JSONL files compatible with
`scripts/experiments/run_mia_experiment.py`.

Usage:
    # Score the parent itself (T == P)
    python scripts/scoring/extract_image_scores.py \\
        --target-model google/vit-base-patch16-224 \\
        --parent-model google/vit-base-patch16-224 \\
        --train-file data/processed/imagenet_shard/train.jsonl \\
        --test-file  data/processed/imagenet_shard/test.jsonl \\
        --output-dir data/scores/imagenet_shard__google_vit_b16_224

    # Score a finetuned target T whose head is different (uses P_head over T_encoder)
    python scripts/scoring/extract_image_scores.py \\
        --target-model ISxOdin/vit-base-oxford-iiit-pets \\
        --parent-model google/vit-base-patch16-224 \\
        --train-file data/processed/imagenet_shard/train.jsonl \\
        --test-file  data/processed/imagenet_shard/test.jsonl \\
        --output-dir data/scores/imagenet_shard__pets

    # Random-init negative control (same architecture as P, fresh weights)
    python scripts/scoring/extract_image_scores.py \\
        --target-model google/vit-base-patch16-224 --random-init --random-seed 0 \\
        --parent-model google/vit-base-patch16-224 \\
        --train-file data/processed/imagenet_shard/train.jsonl \\
        --test-file  data/processed/imagenet_shard/test.jsonl \\
        --output-dir data/scores/imagenet_shard__random_init
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.shard_audit.vit_scores import (
    load_parent_head,
    make_random_init_model,
    score_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _load_image(rec: dict) -> Image.Image:
    p = rec["image_path"]
    if not os.path.isabs(p):
        p = os.path.join(REPO_ROOT, p)
    return Image.open(p).convert("RGB")


def _diagnostics(records: list[dict], score_keys: tuple[str, ...]) -> dict:
    """Compute per-label means and a crude AUC for each score (higher = member)."""
    out: dict = {}
    for k in score_keys:
        vals_pos = [r[k] for r in records if r["label"] == 1]
        vals_neg = [r[k] for r in records if r["label"] == 0]
        if not vals_pos or not vals_neg:
            continue
        # Mann-Whitney U based AUC
        all_vals = [(v, 1) for v in vals_pos] + [(v, 0) for v in vals_neg]
        all_vals.sort()
        ranks = {}
        i = 0
        while i < len(all_vals):
            j = i
            while j + 1 < len(all_vals) and all_vals[j + 1][0] == all_vals[i][0]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed
            for kk in range(i, j + 1):
                ranks[kk] = avg_rank
            i = j + 1
        sum_ranks_pos = sum(ranks[idx] for idx, (_, lbl) in enumerate(all_vals) if lbl == 1)
        n_pos, n_neg = len(vals_pos), len(vals_neg)
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
        auc = u / (n_pos * n_neg)
        out[k] = {
            "mean_label_1": sum(vals_pos) / n_pos,
            "mean_label_0": sum(vals_neg) / n_neg,
            "auc": auc,
            "direction_ok": auc >= 0.5,
            "n_label_1": n_pos,
            "n_label_0": n_neg,
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-image ViT membership scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-model", required=True, help="HF id of the model to audit (T)")
    p.add_argument("--parent-model", required=True, help="HF id of the parent model (P)")
    p.add_argument("--random-init", action="store_true",
                   help="Override target weights with a fresh random initialisation. "
                        "Architecture is taken from --target-model. Useful as a negative control.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-examples", type=int, default=None)
    return p.parse_args()


def _select_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(arg: str, device: str) -> torch.dtype | None:
    if arg == "auto":
        if device == "cpu":
            return torch.float32
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[arg]


def main():
    args = parse_args()
    device = _select_device(args.device)
    dtype = _select_dtype(args.dtype, device)

    logger.info("=== ViT Image Score Extraction ===")
    logger.info("Target:  %s (random_init=%s)", args.target_model, args.random_init)
    logger.info("Parent:  %s", args.parent_model)
    logger.info("Train:   %s", args.train_file)
    logger.info("Test:    %s", args.test_file)
    logger.info("Output:  %s", args.output_dir)
    logger.info("Device:  %s  Dtype: %s", device, dtype)

    parent_head, parent_processor = load_parent_head(args.parent_model, device=device, dtype=dtype)

    if args.random_init:
        logger.info("Initialising target with random weights (seed=%d)", args.random_seed)
        target_model = make_random_init_model(
            args.target_model, seed=args.random_seed, device=device, dtype=dtype,
        )
        target_processor = parent_processor  # share preprocessing
        target_uses_imagenet_head = False
        target_label = f"{args.target_model}::random_init_seed{args.random_seed}"
    else:
        logger.info("Loading target model %s", args.target_model)
        target_model = AutoModelForImageClassification.from_pretrained(
            args.target_model, torch_dtype=dtype
        ).to(device).eval()
        try:
            target_processor = AutoImageProcessor.from_pretrained(args.target_model, use_fast=True)
        except Exception as e:
            logger.warning("No image processor found for target; reusing parent's. (%s)", e)
            target_processor = parent_processor
        target_uses_imagenet_head = (
            isinstance(target_model.classifier, torch.nn.Linear)
            and target_model.classifier.out_features == parent_head.out_features
        )
        target_label = args.target_model

    logger.info("Target uses ImageNet head: %s  (out_features=%s)",
                target_uses_imagenet_head,
                target_model.classifier.out_features
                if isinstance(target_model.classifier, torch.nn.Linear) else "?")

    train_records = _load_jsonl(args.train_file)
    test_records = _load_jsonl(args.test_file)
    if args.max_examples is not None:
        train_records = train_records[: args.max_examples]
        test_records = test_records[: args.max_examples]
    logger.info("Loaded %d train, %d test records", len(train_records), len(test_records))

    score_keys = (
        "p_on_t_logprob",       # primary
        "p_on_t_loss",          # auxiliary (lower = member)
        "t_own_top1_logprob",   # auxiliary
        "t_own_neg_entropy",    # auxiliary
    )

    def _score_split(records, split_name):
        out: list[dict] = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            imgs = [_load_image(r) for r in batch]
            labels = torch.tensor([r["imagenet_class"] for r in batch], dtype=torch.long)
            bundles = score_batch(
                images=imgs,
                labels=labels,
                target_model=target_model,
                parent_head=parent_head,
                image_processor=parent_processor,
                target_processor=target_processor,
                target_uses_imagenet_head=target_uses_imagenet_head,
                device=device,
            )
            for rec, sb in zip(batch, bundles):
                out.append({
                    "id": rec["id"],
                    "label": rec["label"],
                    "phase_split": rec.get("phase_split", split_name),
                    "image_hash": rec["image_hash"],
                    "imagenet_class": rec["imagenet_class"],
                    "model": target_label,
                    "p_on_t_logprob": round(sb.p_on_t_logprob, 6),
                    "p_on_t_loss":    round(sb.p_on_t_loss, 6),
                    "t_own_top1_logprob": round(sb.t_own_top1_logprob, 6),
                    "t_own_neg_entropy":  round(sb.t_own_neg_entropy, 6),
                })
            if (start // args.batch_size + 1) % 5 == 0:
                logger.info("  %s: scored %d/%d", split_name,
                            min(start + args.batch_size, len(records)), len(records))
        return out

    logger.info("Scoring train split...")
    train_scores = _score_split(train_records, "train")
    logger.info("Scoring test split...")
    test_scores = _score_split(test_records, "test")

    train_diag = _diagnostics(train_scores, score_keys)
    test_diag = _diagnostics(test_scores, score_keys)

    logger.info("=== Train diagnostics ===")
    for k, v in train_diag.items():
        logger.info("  %-22s  label0=%+.4f  label1=%+.4f  AUC=%.4f  dir=%s",
                    k, v["mean_label_0"], v["mean_label_1"], v["auc"],
                    "OK" if v["direction_ok"] else "INVERTED")
    logger.info("=== Test diagnostics ===")
    for k, v in test_diag.items():
        logger.info("  %-22s  label0=%+.4f  label1=%+.4f  AUC=%.4f  dir=%s",
                    k, v["mean_label_0"], v["mean_label_1"], v["auc"],
                    "OK" if v["direction_ok"] else "INVERTED")

    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_scores, os.path.join(args.output_dir, "train_scores.jsonl"))
    _write_jsonl(test_scores, os.path.join(args.output_dir, "test_scores.jsonl"))

    manifest = {
        "target_model": args.target_model,
        "target_label": target_label,
        "random_init": args.random_init,
        "random_seed": args.random_seed if args.random_init else None,
        "parent_model": args.parent_model,
        "target_uses_imagenet_head": target_uses_imagenet_head,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "n_train_scored": len(train_scores),
        "n_test_scored": len(test_scores),
        "score_keys": list(score_keys),
        "score_diagnostics_train": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                        for kk, vv in v.items()}
                                    for k, v in train_diag.items()},
        "score_diagnostics_test": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                       for kk, vv in v.items()}
                                   for k, v in test_diag.items()},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest.json written")

    print("\n=== DONE ===")
    print(f"  Output: {args.output_dir}")
    print(f"  train_scores.jsonl: {len(train_scores)}")
    print(f"  test_scores.jsonl:  {len(test_scores)}")


if __name__ == "__main__":
    main()
