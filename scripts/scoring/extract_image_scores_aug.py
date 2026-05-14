"""Augmentation-based black-box membership scoring (full-logit access).

For each image x we generate K augmented copies x_1,...,x_K, push them through
the target model T, and collect K full softmax distributions p_k in R^|classes|.
Members tend to be more stable across augmentations than nonmembers because T's
decision surface has tightened around them. We report several stability-style
scores, all in the convention 'higher = more member-like':

  aug_neg_top1_std        -std_k( max_c p_k[c] )
                          smaller std = more consistent confidence = member.

  aug_neg_logit_var_mean  -mean_c( var_k( logit_k[c] ) )
                          smaller per-class logit variance = member.

  aug_cos_sim_mean        mean_{i<j}  cos( p_i, p_j )
                          higher cosine = more consistent distribution = member.

  aug_neg_kl_mean         -mean_{i<j}  KL( p_i || p_bar ) symmetrised
                          smaller KL = more consistent distribution = member.

The auditor only sees logits over T's class space; no encoder activations or
ground-truth ImageNet labels are used by these scores (logits-only access).

Usage:
    python scripts/scoring/extract_image_scores_aug.py \\
        --target-model MatanBT/vit-base-patch16-224-cifar10 \\
        --train-file data/processed/imagenet_shard/train.jsonl \\
        --test-file  data/processed/imagenet_shard/test.jsonl \\
        --output-dir data/scores_aug/imagenet_shard__cifar10 \\
        --n-aug 8 --batch-size 16
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.shard_audit.vit_scores import make_random_init_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records, path):
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


def make_augmenter(image_size: int = 224, image_mean=None, image_std=None) -> transforms.Compose:
    """Random augmentation pipeline applied to PIL images, returning a normalised tensor.
    Designed to mirror common ViT augmentations: small random crops + flip + colour jitter."""
    if image_mean is None: image_mean = [0.5, 0.5, 0.5]
    if image_std is None:  image_std  = [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.10), antialias=True),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])


@torch.no_grad()
def aug_score_batch(
    images: list[Image.Image],
    target_model,
    augmenter: transforms.Compose,
    n_aug: int,
    device: str,
    dtype: torch.dtype,
) -> list[dict]:
    """For each image generate n_aug augmented views, forward through T,
    and return per-image stability statistics."""
    target_model.eval()
    B = len(images)
    K = n_aug
    # Build (B*K) augmented tensors
    tensors = []
    for img in images:
        for _ in range(K):
            tensors.append(augmenter(img))
    x = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)  # [B*K, 3, H, W]
    out = target_model(pixel_values=x)
    logits = out.logits  # [B*K, C]
    log_p = F.log_softmax(logits, dim=-1)
    probs = log_p.exp()

    C = logits.shape[-1]
    logits_bk = logits.view(B, K, C).float()                              # [B, K, C]
    log_p_bk  = log_p.view(B, K, C).float()
    probs_bk  = probs.view(B, K, C).float()

    # Top-1 softmax probability per augmentation
    top1_p = probs_bk.max(dim=-1).values              # [B, K]
    top1_std = top1_p.std(dim=-1, unbiased=False)     # [B]

    # Per-class logit variance across augmentations, then mean over classes
    logit_var_mean = logits_bk.var(dim=1, unbiased=False).mean(dim=-1)  # [B]

    # Pairwise cosine similarity of softmax distributions
    p_norm = F.normalize(probs_bk, p=2, dim=-1)        # [B, K, C]
    cos_mat = torch.einsum("bkc,bjc->bkj", p_norm, p_norm)  # [B, K, K]
    iu = torch.triu_indices(K, K, offset=1, device=cos_mat.device)
    cos_pairs = cos_mat[:, iu[0], iu[1]]               # [B, K*(K-1)/2]
    cos_mean = cos_pairs.mean(dim=-1)                  # [B]

    # Symmetrised KL between each augmented distribution and the per-image mean
    p_bar = probs_bk.mean(dim=1, keepdim=True)         # [B, 1, C]
    log_p_bar = (p_bar + 1e-12).log()                  # [B, 1, C]
    # KL(p_k || p_bar) per augmentation, mean over K
    kl_per = (probs_bk * (log_p_bk - log_p_bar)).sum(dim=-1)   # [B, K]
    kl_mean = kl_per.mean(dim=-1)                              # [B]

    out: list[dict] = []
    for b in range(B):
        out.append({
            "aug_neg_top1_std":       float(-top1_std[b].item()),
            "aug_neg_logit_var_mean": float(-logit_var_mean[b].item()),
            "aug_cos_sim_mean":       float(cos_mean[b].item()),
            "aug_neg_kl_mean":        float(-kl_mean[b].item()),
        })
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Augmentation-based black-box MIA scoring with full-logit access.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-model", required=True)
    p.add_argument("--random-init", action="store_true",
                   help="Override target weights with a fresh random initialisation.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-aug", type=int, default=8,
                   help="Augmented views per image")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Image batch size; effective batch fed to T is batch_size * n_aug")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _select_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(arg: str, device: str) -> torch.dtype:
    if arg == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[arg]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    dtype = _select_dtype(args.dtype, device)

    logger.info("=== Augmentation-based MIA scoring ===")
    logger.info("Target:    %s (random_init=%s)", args.target_model, args.random_init)
    logger.info("n_aug:     %d", args.n_aug)
    logger.info("batch:     %d (effective fwd batch = %d)", args.batch_size, args.batch_size * args.n_aug)
    logger.info("device:    %s  dtype: %s", device, dtype)
    logger.info("Output:    %s", args.output_dir)

    if args.random_init:
        target_model = make_random_init_model(
            args.target_model, seed=args.random_seed, device=device, dtype=dtype,
        )
        target_label = f"{args.target_model}::random_init_seed{args.random_seed}"
    else:
        target_model = AutoModelForImageClassification.from_pretrained(
            args.target_model, torch_dtype=dtype,
        ).to(device).eval()
        target_label = args.target_model

    # Use the target's image processor mean/std (or parent's as a default)
    try:
        proc = AutoImageProcessor.from_pretrained(args.target_model, use_fast=True)
        mean = proc.image_mean
        std = proc.image_std
        size = (proc.size.get("height") or proc.size.get("shortest_edge")
                or proc.crop_size.get("height", 224))
    except Exception:
        mean, std, size = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 224
    augmenter = make_augmenter(image_size=int(size), image_mean=mean, image_std=std)
    logger.info("Augmenter: size=%s mean=%s std=%s", size, mean, std)

    train_records = _load_jsonl(args.train_file)
    test_records = _load_jsonl(args.test_file)
    if args.max_examples is not None:
        train_records = train_records[: args.max_examples]
        test_records = test_records[: args.max_examples]
    logger.info("Loaded %d train, %d test records", len(train_records), len(test_records))

    score_keys = (
        "aug_neg_top1_std",
        "aug_neg_logit_var_mean",
        "aug_cos_sim_mean",
        "aug_neg_kl_mean",
    )

    def _score_split(records, split_name):
        out: list[dict] = []
        n_done = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            imgs = [_load_image(r) for r in batch]
            stats = aug_score_batch(
                imgs, target_model, augmenter,
                n_aug=args.n_aug, device=device, dtype=dtype,
            )
            for rec, s in zip(batch, stats):
                out.append({
                    "id": rec["id"],
                    "label": rec["label"],
                    "phase_split": rec.get("phase_split", split_name),
                    "image_hash": rec["image_hash"],
                    "imagenet_class": rec["imagenet_class"],
                    "model": target_label,
                    **{k: round(s[k], 6) for k in score_keys},
                })
            n_done += len(batch)
            if (start // args.batch_size + 1) % 10 == 0:
                logger.info("  %s: scored %d/%d", split_name, n_done, len(records))
        return out

    logger.info("Scoring train split...")
    train_scores = _score_split(train_records, "train")
    logger.info("Scoring test split...")
    test_scores = _score_split(test_records, "test")

    def _diag(records):
        d = {}
        for k in score_keys:
            pos = [r[k] for r in records if r["label"] == 1]
            neg = [r[k] for r in records if r["label"] == 0]
            # Mann-Whitney AUC
            all_v = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
            ranks = {}
            i = 0
            while i < len(all_v):
                j = i
                while j + 1 < len(all_v) and all_v[j + 1][0] == all_v[i][0]:
                    j += 1
                avg = (i + j) / 2 + 1
                for kk in range(i, j + 1):
                    ranks[kk] = avg
                i = j + 1
            sumr = sum(ranks[idx] for idx, (_, lbl) in enumerate(all_v) if lbl == 1)
            n_pos, n_neg = len(pos), len(neg)
            u = sumr - n_pos * (n_pos + 1) / 2
            auc = u / (n_pos * n_neg) if n_pos and n_neg else 0.5
            d[k] = {"mean_label_1": sum(pos) / max(n_pos, 1),
                    "mean_label_0": sum(neg) / max(n_neg, 1),
                    "auc": auc, "direction_ok": auc >= 0.5}
        return d

    train_diag = _diag(train_scores)
    test_diag = _diag(test_scores)
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
        "n_aug": args.n_aug,
        "batch_size": args.batch_size,
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

    print("\n=== DONE ===")
    print(f"  Output: {args.output_dir}")
    print(f"  train_scores.jsonl: {len(train_scores)}")
    print(f"  test_scores.jsonl:  {len(test_scores)}")


if __name__ == "__main__":
    main()
