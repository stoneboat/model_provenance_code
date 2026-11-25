"""Train the three published witness-sharing histories and audit the final targets."""

import argparse
from pathlib import Path
import subprocess
import sys

from experiments.watermark_inheritance.common import load_json, write_json
from experiments.watermark_inheritance.prepare import file_sha256
from experiments.watermark_inheritance.runners.m7vv_confirm_shards import RUN


def run(run_dir):
    protocol = load_json(run_dir / "protocol.json")
    files = load_json(run_dir / "prepared/shards.manifest.json")["files"]
    trainers = "experiments.watermark_inheritance.runners.minimal_q4_train"
    score = "experiments.watermark_inheritance.score"
    lineages = {}
    for arm, branch in protocol["branches"].items():
        witness = "s_double_prime_train" if arm == "C" else "s_train"
        if file_sha256(run_dir / "prepared" / f"{witness}.jsonl") != files[witness]:
            raise ValueError(f"Training projection changed for arm {arm}")
        for stage in ("parent", "target"):
            subprocess.run([sys.executable, "-m", trainers, arm, stage, "--run-dir", str(run_dir)], check=True)
        branch_dir = run_dir / "train" / arm
        parent = load_json(branch_dir / "lineage.json")
        target = load_json(branch_dir / "dolly/lineage.json")
        if (parent["data_sha256"] != files[witness]
                or parent["weights_sha256"] != target["source_sha256"]
                or parent["updates"] != 567 or target["updates"] != 16
                or parent["seed"] != branch["model_seed"] or target["seed"] != branch["model_seed"]):
            raise ValueError(f"Training lineage failed verification for arm {arm}")
        lineages[arm] = {"parent": parent, "target": target}
        subprocess.run([sys.executable, "-m", score, "--run-dir", str(run_dir),
                        "--checkpoint-name", arm, "--checkpoint-path", target["checkpoint"]], check=True)
    if len({arm["target"]["data_sha256"] for arm in lineages.values()}) != 1:
        raise ValueError("Downstream data differ across arms")
    if len({stage["weights_sha256"] for arm in lineages.values() for stage in arm.values()}) != 6:
        raise ValueError("Independent histories must have distinct parent and target weights")
    write_json(run_dir / "lineages.json", lineages)
    subprocess.run(
        [sys.executable, "-m", "experiments.watermark_inheritance.matched_process_audit",
         "--run-dir", str(run_dir), "--output-dir", str(run_dir / "audit")],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
