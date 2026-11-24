"""Run the two Pile controls and the selected Pythia families on one shared split."""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_audit import FAMILIES, MIMIR_REVISION, REPO_ROOT, SCORES, pinned_revision


def run(script, *args):
    subprocess.run([sys.executable, script, *args], cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Use existing train.jsonl and test.jsonl")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=["1.4b"])
    args = parser.parse_args()

    root = REPO_ROOT / "artifacts/pile_controls_reconstruction"
    if args.data_dir is None:
        data = root / "seed_0/prepared/mimir_github"
        run("scripts/data/prepare_mimir_domain.py", "--dataset-revision", MIMIR_REVISION,
            "--config", "github", "--split", "ngram_13_0.2", "--seed", "0",
            "--min-words", "8", "--max-words", "64", "--num-train-member", "560",
            "--num-train-nonmember", "340", "--num-test-per-class", "400",
            "--output-dir", data)
    else:
        data = args.data_dir.resolve()
    train, test = data / "train.jsonl", data / "test.jsonl"
    for path in (train, test):
        if not path.is_file():
            parser.error(f"Missing prepared split: {path}")

    models = [
        ("initialization", "EleutherAI/pythia-1.4b", "ae04bdc1923b877cf65aee9da264d30555f011ab"),
        ("independent", "EleutherAI/gpt-neo-1.3B", "dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8"),
    ]
    for key in args.families:
        family = FAMILIES[key]
        models.append(("parent", family.parent, pinned_revision(family.parent)))
        for target in family.targets:
            if target == "nnheui/pythia-1.4b-sft-full":
                print(f"Skipping unavailable Table 6 checkpoint {target}", flush=True)
                continue
            models.append(("derivative", target, pinned_revision(target)))

    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    print(f"Data: {data}\nResults: {output}", flush=True)
    with (output / "results.csv").open("w", newline="") as handle:
        table = csv.writer(handle)
        table.writerow(["role", "model", "revision", "score", "fitted_threshold",
                        "eval_members", "eval_controls", "signed_advantage",
                        "absolute_advantage", "reference_cutoff_0.05", "detected"])
        for role, model, revision in models:
            dest = output / f"{role}_{model.replace('/', '__')}"
            run("scripts/scoring/score_causal_lm_logprobs.py", "--model", model,
                "--model-revision", revision, "--train-file", train, "--test-file", test,
                "--min-k-pcts", "20", "--batch-size", "4", "--max-length", "512",
                "--dtype", "float16", "--device", "cuda", "--debug-examples", "0",
                "--output-dir", dest)
            run("scripts/audit/run_threshold_audit.py", "--model-label", model,
                "--train-scores", dest / "train_scores.jsonl", "--test-scores",
                dest / "test_scores.jsonl", "--score-keys", SCORES, "--output-dir", dest)
            for result in json.loads((dest / "results.json").read_text())["main_results"]:
                metrics = result["test"]
                pos, neg = metrics["n_pos"], metrics["n_neg"]
                gap = metrics["tp"] / pos - metrics["fp"] / neg
                cutoff = math.sqrt(0.5 * (1 / pos + 1 / neg) * math.log(2 / 0.05))
                table.writerow([role, model, revision, result["score_name"],
                                result["calibrated_threshold"], pos, neg, gap, abs(gap),
                                cutoff, abs(gap) > cutoff])
            handle.flush()


if __name__ == "__main__":
    main()
