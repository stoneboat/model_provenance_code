"""Train one arm of the nine-epoch watermark witness-sharing experiment."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from experiments.watermark_inheritance.common import load_json, write_json
from experiments.watermark_inheritance.train import train_stage

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(run, config, witness, prepared_inputs):
    prepared = run / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for name, source in [("pairs.jsonl", witness), ("dolly.jsonl",prepared_inputs / "dolly.jsonl"),
                         ("dolly_manifest.json",prepared_inputs / "dolly_manifest.json")]:
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
    publication = args.run_dir
    protocol = load_json(publication / "protocol.json")
    generation = load_json(publication / "config.json")
    config = dict(protocol["training"])
    epoch = config.pop("witness_epochs")
    downstream = config.pop("downstream_epochs")
    witness_lr = config.pop("witness_learning_rate")
    cycle_epochs = config.pop("lr_cycle_epochs")
    seed = protocol["branches"][args.arm]["model_seed"]
    config.update(seed=seed, model_seed=seed, arm=args.arm,
                  model_path=generation["model_path"], tokenizer_path=generation["tokenizer_path"],
                  stage_epochs={"parent": epoch, "target": downstream})
    branch = publication / "train" / args.arm
    parent = branch / "checkpoints" / "parent"
    run = branch
    if args.stage == "parent":
        config.update(learning_rate=witness_lr, lr_cycle_epochs=cycle_epochs)
    else:
        config.update(parent_checkpoint=str(parent.resolve()), parent_epoch=epoch,
                      parent_weight_sha256=digest(parent / "model.safetensors"))
        run = branch / "dolly"
    prepared_inputs = publication / "prepared"
    witness_name = "s_double_prime_train.jsonl" if args.arm == "C" else "s_train.jsonl"
    prepare(run, config, prepared_inputs / witness_name, prepared_inputs)
    start = time.monotonic()
    begin_utc = datetime.now(timezone.utc).isoformat()
    stats = train_stage(run, args.stage, "cuda:0")
    final = parent if args.stage == "parent" else run/"checkpoints/target_epoch1"
    log = dict(stage=args.stage, prefix_epoch=epoch, checkpoint=str(final), weights_sha256=digest(final/"model.safetensors"),
               start_utc=begin_utc, wall_seconds=time.monotonic()-start, updates=stats["global_step"], seed=seed,
               source_sha256=digest(Path(config["model_path"])/"model.safetensors") if args.stage=="parent" else config["parent_weight_sha256"],
               data_sha256=stats["identity"]["data_sha256"])
    write_json(run / "lineage.json", log)
    print(json.dumps(log))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["A", "B", "C"])
    parser.add_argument("stage", choices=["parent", "target"])
    parser.add_argument("--run-dir", type=Path, required=True)
    run_stage(parser.parse_args())
