"""Prepare ImageNet member/nonmember dataset for ViT shard-membership auditing.

Dataset: evanarlian/imagenet_1k_resized_256 (non-gated mirror of ILSVRC/imagenet-1k,
images resized so the shorter side is 256). Same images, same labels as the original
ImageNet-1k -> training samples are members of any model trained on ImageNet-1k,
validation samples are non-members.

CRITICAL: class-balanced sampling.
The HF parquet shards in this dataset are class-sorted: train shard 0 contains
only ~19 of the 1000 classes, while a val shard contains ~25 images of every
class. Naive uniform sampling therefore yields member/nonmember sets with
wildly different class distributions, which would let any classifier "detect
membership" purely by class prior. We avoid this by:
  1. Loading the requested train shards and val shards.
  2. Restricting to the intersection of classes covered by both.
  3. For each retained class, sampling exactly N member images (from train
     shards) and N nonmember images (from val shards).

Output directory structure mirrors the LLM pipeline:
    <output_dir>/train.jsonl      MIA calibration split (member + nonmember)
    <output_dir>/test.jsonl       MIA evaluation split (member + nonmember)
    <output_dir>/images/<id>.jpg  one file per record
    <output_dir>/manifest.json

JSONL record schema:
    {
      "id": "member-00042",
      "label": 1,                       # 1 = member (train shard), 0 = nonmember (val shard)
      "phase_split": "train" | "test",  # MIA calibration vs evaluation split
      "split_origin": "member" | "nonmember",
      "imagenet_class": 217,            # 0..999, the ground-truth ImageNet class
      "image_path": "data/processed/.../images/member-00042.jpg",
      "image_hash": "sha256:..."
    }

Usage:
    # Class-balanced: m_per_class members + m_per_class nonmembers per class,
    # 50/50 split into MIA train (calibration) and MIA test (evaluation).
    python scripts/data/prepare_imagenet_shard.py \\
        --train-shard-idxs 0,1 --val-shard-idxs 0,1 \\
        --members-per-class 50 --nonmembers-per-class 50 \\
        --train-frac 0.5 --seed 0 \\
        --output-dir data/processed/imagenet_shard
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DATASET_REPO = "evanarlian/imagenet_1k_resized_256"
TRAIN_SHARD_TEMPLATE = "data/train-{idx:05d}-of-00052-"
VAL_SHARD_TEMPLATE = "data/val-{idx:05d}-of-00002-"


def _list_dataset_files(token: str | None = None) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(DATASET_REPO, token=token)
    return [s.rfilename for s in info.siblings]


def _resolve_shard_filename(prefix: str, all_files: list[str]) -> str:
    matches = [f for f in all_files if f.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No file starting with {prefix!r} in dataset {DATASET_REPO}")
    return matches[0]


def _load_shard_table(filename: str, token: str | None = None):
    logger.info("Downloading parquet shard: %s", filename)
    local_path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=filename, token=token
    )
    logger.info("  -> %s", local_path)
    return pq.read_table(local_path)


def _decode_image_bytes(payload) -> Image.Image:
    """The HF parquet schema stores images as a struct {'bytes': binary, 'path': string}."""
    if isinstance(payload, dict):
        b = payload.get("bytes")
    else:
        b = payload
    if b is None:
        raise ValueError("Empty image payload in parquet row")
    return Image.open(io.BytesIO(b)).convert("RGB")


def _sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _save_jpeg(img: Image.Image, path: str, quality: int = 90) -> bytes:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return data


def _parse_idx_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _group_indices_by_class(label_col) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i in range(len(label_col)):
        c = int(label_col[i].as_py())
        out.setdefault(c, []).append(i)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare ImageNet-1k member/nonmember dataset for ViT auditing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-shard-idxs", default="0",
                   help="Comma-separated train parquet shard indices (0..51)")
    p.add_argument("--val-shard-idxs", default="0,1",
                   help="Comma-separated val parquet shard indices (0..1). Default uses both.")
    p.add_argument("--members-per-class", type=int, default=50,
                   help="Member images sampled per ImageNet class (from train shards)")
    p.add_argument("--nonmembers-per-class", type=int, default=50,
                   help="Nonmember images sampled per ImageNet class (from val shards)")
    p.add_argument("--train-frac", type=float, default=0.5,
                   help="Fraction of each per-class subset routed to MIA-calibration (train.jsonl); "
                        "rest goes to MIA-evaluation (test.jsonl)")
    p.add_argument("--max-classes", type=int, default=None,
                   help="If set, randomly subselect at most this many of the classes that "
                        "appear in *both* train and val shards.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--output-dir", default="data/processed/imagenet_shard")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--list-files", action="store_true", help="List dataset files and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_files:
        for f in _list_dataset_files(token=args.token):
            print(f)
        return

    train_idxs = _parse_idx_list(args.train_shard_idxs)
    val_idxs = _parse_idx_list(args.val_shard_idxs)

    logger.info("=== ImageNet ViT Shard Preparation (class-balanced) ===")
    logger.info("Dataset:        %s", DATASET_REPO)
    logger.info("Train shards:   %s", train_idxs)
    logger.info("Val shards:     %s", val_idxs)
    logger.info("Per class:      %d members + %d nonmembers",
                args.members_per_class, args.nonmembers_per_class)
    logger.info("Train frac:     %.2f", args.train_frac)
    logger.info("Max classes:    %s", args.max_classes)
    logger.info("Seed:           %d", args.seed)
    logger.info("Output dir:     %s", args.output_dir)

    all_files = _list_dataset_files(token=args.token)

    import pyarrow as pa
    train_tables = []
    for ti in train_idxs:
        fn = _resolve_shard_filename(TRAIN_SHARD_TEMPLATE.format(idx=ti), all_files)
        train_tables.append(_load_shard_table(fn, token=args.token))
    val_tables = []
    for vi in val_idxs:
        fn = _resolve_shard_filename(VAL_SHARD_TEMPLATE.format(idx=vi), all_files)
        val_tables.append(_load_shard_table(fn, token=args.token))
    train_table = pa.concat_tables(train_tables)
    val_table = pa.concat_tables(val_tables)
    logger.info("Loaded %d total train rows, %d total val rows", train_table.num_rows, val_table.num_rows)

    train_by_class = _group_indices_by_class(train_table.column("label"))
    val_by_class = _group_indices_by_class(val_table.column("label"))
    common_classes = sorted(set(train_by_class) & set(val_by_class))
    logger.info("Classes covered:  train=%d, val=%d, intersection=%d",
                len(train_by_class), len(val_by_class), len(common_classes))
    if not common_classes:
        raise SystemExit("No class appears in both train and val shards. "
                         "Pass more --train-shard-idxs.")

    rng = random.Random(args.seed)
    if args.max_classes is not None and len(common_classes) > args.max_classes:
        common_classes = sorted(rng.sample(common_classes, args.max_classes))
    logger.info("Using %d classes (range [%d..%d])", len(common_classes),
                min(common_classes), max(common_classes))

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_records: list[dict] = []
    test_records: list[dict] = []
    seen_hashes: set[str] = set()
    n_dup = 0

    def _materialise_class(table, idx_list: list[int], role: str, target: int, class_id: int):
        nonlocal n_dup
        label = 1 if role == "member" else 0
        avail = list(idx_list)
        sample_rng = random.Random((args.seed if role == "member" else args.seed + 1) * 1000003 + class_id)
        sample_rng.shuffle(avail)
        chosen = avail[:target]
        n_train = int(round(args.train_frac * len(chosen)))
        # Decide MIA train/test routing for this (class, role) cell deterministically
        per_class_train = chosen[:n_train]
        per_class_test = chosen[n_train:]

        img_col = table.column("image")
        for kind, indices in (("train", per_class_train), ("test", per_class_test)):
            for j, idx in enumerate(indices):
                payload = img_col[idx].as_py()
                try:
                    img = _decode_image_bytes(payload)
                except Exception as e:
                    logger.warning("Skipping %s class=%d row=%d: decode failed (%s)", role, class_id, idx, e)
                    continue
                rec_id = f"{role}-c{class_id:04d}-{kind}-{j:04d}"
                out_path = os.path.join(images_dir, f"{rec_id}.jpg")
                data = _save_jpeg(img, out_path, quality=args.jpeg_quality)
                h = _sha256(data)
                if h in seen_hashes:
                    n_dup += 1
                seen_hashes.add(h)
                rec = {
                    "id": rec_id,
                    "label": label,
                    "phase_split": kind,
                    "split_origin": role,
                    "imagenet_class": class_id,
                    "image_path": os.path.relpath(out_path, REPO_ROOT),
                    "image_hash": h,
                }
                if kind == "train":
                    train_records.append(rec)
                else:
                    test_records.append(rec)
        return len(per_class_train), len(per_class_test)

    n_mem_train = n_mem_test = n_non_train = n_non_test = 0
    classes_dropped = []
    for c in common_classes:
        avail_mem = len(train_by_class[c])
        avail_non = len(val_by_class[c])
        if avail_mem < args.members_per_class or avail_non < args.nonmembers_per_class:
            classes_dropped.append((c, avail_mem, avail_non))
            continue
        kt, ke = _materialise_class(train_table, train_by_class[c], "member",
                                    args.members_per_class, c)
        n_mem_train += kt; n_mem_test += ke
        kt, ke = _materialise_class(val_table, val_by_class[c], "nonmember",
                                    args.nonmembers_per_class, c)
        n_non_train += kt; n_non_test += ke

    if classes_dropped:
        logger.warning("Dropped %d classes for insufficient samples: %s",
                       len(classes_dropped), classes_dropped[:5])

    logger.info("Materialised: %d member-train, %d member-test, %d nonmember-train, %d nonmember-test",
                n_mem_train, n_mem_test, n_non_train, n_non_test)
    logger.info("Cross-split duplicate hashes: %d", n_dup)

    rng.shuffle(train_records)
    rng.shuffle(test_records)

    # Hash overlap between MIA train/test
    train_hashes = {r["image_hash"] for r in train_records}
    test_hashes = {r["image_hash"] for r in test_records}
    tt_overlap = len(train_hashes & test_hashes)
    if tt_overlap:
        logger.warning("MIA train/test image-hash overlap: %d", tt_overlap)
    else:
        logger.info("MIA train/test hash overlap: 0")

    def _write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %d records to %s", len(records), path)

    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records, os.path.join(args.output_dir, "test.jsonl"))

    manifest = {
        "dataset_repo": DATASET_REPO,
        "train_shard_idxs": train_idxs,
        "val_shard_idxs": val_idxs,
        "members_per_class": args.members_per_class,
        "nonmembers_per_class": args.nonmembers_per_class,
        "train_frac": args.train_frac,
        "n_classes_used": len(common_classes) - len(classes_dropped),
        "n_classes_dropped": len(classes_dropped),
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "n_train_member": n_mem_train,
        "n_train_nonmember": n_non_train,
        "n_test_member": n_mem_test,
        "n_test_nonmember": n_non_test,
        "seed": args.seed,
        "jpeg_quality": args.jpeg_quality,
        "mia_train_test_image_hash_overlap": tt_overlap,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(args.output_dir),
        "label_assignment": {
            "1": "member (sampled from ImageNet train shard)",
            "0": "nonmember (sampled from ImageNet val shard)",
        },
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest.json written")

    print("\n=== DONE ===")
    print(f"  Output: {args.output_dir}")
    print(f"  train.jsonl: {len(train_records)} records ({n_mem_train} mem / {n_non_train} non)")
    print(f"  test.jsonl:  {len(test_records)} records ({n_mem_test} mem / {n_non_test} non)")


if __name__ == "__main__":
    main()
