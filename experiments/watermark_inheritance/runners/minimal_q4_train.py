"""Train one fixed witness-budget family, or its one-epoch Dolly branch at a saved prefix."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from experiments.watermark_inheritance.common import load_json, write_json
from experiments.watermark_inheritance.train import train_stage

TEMPLATE = Path("artifacts/m7vv_q4/train/A")
FAMILIES = {
    "bf16_lr1": ("fp32_master_bf16_autocast", 1e-5),
    "bf16_lr5": ("fp32_master_bf16_autocast", 5e-5),
    "head32_lr5": ("fp32_master_bf16_backbone_fp32_head", 5e-5),
}
PREFIXES = [1, 2, 3, 4, 6, 9, 12, 18, 24]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(run, config, witness):
    prepared = run / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for name, source in [("pairs.jsonl", witness), ("dolly.jsonl",TEMPLATE / "prepared/dolly.jsonl"),
                         ("dolly_manifest.json",TEMPLATE / "prepared/dolly_manifest.json")]:
        target = prepared / name
        if target.exists():
            if target.resolve() != source.resolve():
                raise ValueError(f"Existing input differs from requested input: {target}")
        else:
            target.symlink_to(source.resolve(strict=True))
    config_file = run / "config.json"
    if config_file.exists() and load_json(config_file) != config:
        raise ValueError(f"Config changed in {run}")
    write_json(config_file, config)


def run_stage(args):
    if args.epoch not in PREFIXES:
        raise ValueError(f"Expected one of {PREFIXES}")
    config = load_json(TEMPLATE / "config.json")
    config["seed"] = config["model_seed"] = args.seed
    config["arm"] = args.family
    run = args.root / args.family
    parent = run / "checkpoints" / ("parent" if args.epoch == max(PREFIXES) else f"parent_epoch{args.epoch}")
    if args.stage == "parent":
        precision, lr = FAMILIES[args.family]
        save_epochs = getattr(args, "checkpoint_epochs", None) or PREFIXES
        if getattr(args, "save_only_requested", False):
            save_epochs = [args.epoch]
        config.update(precision=precision, learning_rate=lr, lr_cycle_epochs=3, save_epochs=save_epochs)
        config["stage_epochs"]["parent"] = max(PREFIXES)
    else:
        config.update(parent_checkpoint=str(parent.resolve()), parent_epoch=args.epoch,
                      parent_weight_sha256=digest(parent/"model.safetensors"))
        run = run / f"dolly_epoch{args.epoch}"
    prepare(run, config, args.witness)
    start = time.monotonic()
    begin_utc = datetime.now(timezone.utc).isoformat()
    stats = train_stage(run, args.stage, "cuda:0", stop_epoch=args.epoch if args.stage=="parent" else 1)
    final = parent if args.stage == "parent" else run/"checkpoints/target_epoch1"
    log = dict(stage=args.stage, prefix_epoch=args.epoch, checkpoint=str(final), weights_sha256=digest(final/"model.safetensors"),
               start_utc=begin_utc, wall_seconds=time.monotonic()-start, updates=stats["global_step"], seed=args.seed,
               source_sha256=digest(Path(config["model_path"])/"model.safetensors") if args.stage=="parent" else config["parent_weight_sha256"],
               data_sha256=stats["identity"]["data_sha256"])
    write_json(run / f"ledger_{args.stage}_epoch{args.epoch}.json", log)
    print(json.dumps(log))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("stage", choices=["parent","target"])
    parser.add_argument("epoch", type=int)
    parser.add_argument("--root",type=Path,default=Path("artifacts/m7vv_minimal/search"))
    parser.add_argument("--witness",type=Path,default=TEMPLATE/"prepared/pairs.jsonl")
    parser.add_argument("--seed",type=int,default=34711)
    parser.add_argument("--checkpoint-epochs", nargs="+", type=int, choices=PREFIXES,
                        help="Fixed checkpoint boundaries across resumed calls on a common trajectory")
    parser.add_argument("--save-only-requested", action="store_true",
                        help="Keep only the selected witness prefix for independent confirmation")
    run_stage(parser.parse_args())
