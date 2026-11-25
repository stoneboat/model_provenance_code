"""Run the six frozen-split public audits for responses qmGv Q1 and Q3."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download
import yaml

from lamini_data import SPLIT_FILES


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
CONFIG = PACKAGE / "configs/rebuttal_public_controls.json"
PROTOCOL = PACKAGE / "configs/matched_lamini.yaml"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def score(label, model, revision, tokenizer_model, settings, data, output):
    inference = settings["inference"]
    flags = ["--model-revision", revision] if model == tokenizer_model else []
    execute("scripts/scoring/score_causal_lm_logprobs.py",
            "--model", model, *flags, "--tokenizer-model", tokenizer_model, "--tokenizer-revision", revision,
            "--train-file", data / "train.jsonl", "--test-file", data / "test.jsonl",
            "--dtype", inference["dtype"], "--batch-size", inference["batch_size"],
            "--max-length", inference["max_length"], "--device", "cuda:0", "--min-k-pcts", "20",
            "--debug-examples", "0", "--output-dir", output)
    audit = settings["audit"]
    execute("scripts/audit/run_threshold_audit.py",
            "--model-label", label, "--train-scores", output / "train_scores.jsonl",
            "--test-scores", output / "test_scores.jsonl", "--output-dir", output,
            "--score-keys", "min_k_20_logprob,mean_logprob", "--primary-score", "min_k_20_logprob",
            "--n-thresholds", audit["n_thresholds"], "--criterion", audit["criterion"])


def lamini_data(protocol, target):
    spec = protocol["training_data"]
    for filename, _ in SPLIT_FILES.values():
        path = hf_hub_download(spec["dataset"], filename, subfolder="data",
                               revision=spec["revision"], repo_type="dataset", local_dir=target)
        if digest(Path(path)) != spec["parquet"][filename]:
            raise ValueError(f"Lamini source hash differs for {filename}")
    return target / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimir-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.mimir_dir.resolve()
    output = args.output_dir.resolve()
    settings = json.loads(CONFIG.read_text())
    for filename, expected in settings["frozen_split_sha256"].items():
        if digest(data_dir / filename) != expected:
            raise ValueError(f"{filename} does not match the frozen MIMIR split")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    data = lamini_data(protocol, output / "lamini_data")
    for index, (label, model) in enumerate(settings["base_models"].items(), start=1):
        model_id, revision = model["id"], model["revision"]
        dest = output / f"model_{index}"
        score(label, model_id, revision, model_id, settings, data_dir, dest / "base")
        if label in settings["training_model_keys"]:
            execute(PACKAGE / "runners/train_matched_lamini.py",
                    "--model", settings["training_model_keys"][label], "--output-dir", dest / "checkpoint",
                    "--protocol", PROTOCOL, "--data-dir", data,
                    "--construction", data_dir / "train.jsonl", "--base-scores", dest / "base/train_scores.jsonl")
            score(label + " + Lamini", dest / "checkpoint/merged", revision,
                  model_id, settings, data_dir, dest / "lamini")


if __name__ == "__main__":
    main()
