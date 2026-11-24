"""Run the prespecified MIMIR GitHub audit and paired contrasts on four score caches."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
PYTHIA_CACHE = (
    REPO / "artifacts/pile_controls_reconstruction/20261124T160154122852Z"
    / "parent_EleutherAI__pythia-1.4b"
)
SCORES = ("min_k_20_logprob", "mean_logprob")
ROW_KEY = ("id", "label", "phase_split", "text_hash")


def load_rows(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return tuple(row[key] for key in ROW_KEY)


def verify_scores(score_path, expected_rows):
    manifest = json.loads((score_path / "manifest.json").read_text())
    for key, expected in (
        ("batch_size", 4), ("max_length", 512), ("dtype", "torch.float16"),
        ("quantization", "none"), ("n_train_scored", 1000), ("n_test_scored", 400),
    ):
        if manifest[key] != expected:
            raise ValueError(f"{score_path}: expected {key}={expected}, found {manifest[key]}")
    if 20 not in manifest["k_pcts"]:
        raise ValueError(f"Missing MIN-K20 in {score_path}")
    token_counts = {}
    for split in ("train", "test"):
        rows = load_rows(score_path / f"{split}_scores.jsonl")
        if [identity(row) for row in rows] != [identity(row) for row in expected_rows[split]]:
            raise ValueError(f"Scored row identities differ from the frozen {split} split at {score_path}")
        if any(not math.isfinite(row[score]) for score in SCORES for row in rows):
            raise ValueError(f"Invalid {split} score in {score_path}")
        token_counts[split] = [(row["num_input_tokens"], row["num_scored_tokens"]) for row in rows]
        if split == "test":
            test_rows = rows
    return test_rows, token_counts


def fit_audit(score_path, audit_dir, model_label):
    command = [
        sys.executable, str(REPO / "scripts/audit/run_threshold_audit.py"),
        "--train-scores", str(score_path / "train_scores.jsonl"),
        "--test-scores", str(score_path / "test_scores.jsonl"),
        "--output-dir", str(audit_dir), "--model-label", model_label,
        "--score-keys", ",".join(SCORES), "--primary-score", SCORES[0],
        "--criterion", "balanced_accuracy", "--n-thresholds", "2000",
        "--run-shuffled-control", "--shuffled-seed", "42",
    ]
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "audit.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((audit_dir / "results.json").read_text())


def bootstrap_contrasts(decisions, labels):
    # Both sets of draws are shared by every model and score.
    rng = np.random.default_rng(3324)
    member, control = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    member_draws = rng.integers(len(member), size=(10000, len(member)))
    control_draws = rng.integers(len(control), size=(10000, len(control)))
    comparisons = [
        ("Pythia adaptation minus base", "pythia_adapted", "pythia_base"),
        ("BioGPT adaptation minus base", "biogpt_adapted", "biogpt_base"),
        ("Adapted Pythia minus adapted BioGPT", "pythia_adapted", "biogpt_adapted"),
    ]
    rows = []
    for comparison, first, second in comparisons:
        for score in SCORES:
            difference = decisions[first, score].astype(np.int8) - decisions[second, score]
            member_diff, control_diff = difference[member], difference[control]
            delta = (member_diff.sum() - control_diff.sum()) / len(member)
            draws = (member_diff[member_draws].sum(axis=1)
                     - control_diff[control_draws].sum(axis=1)) / len(member)
            low, high = np.percentile(draws, [2.5, 97.5])
            rows.append(dict(comparison=comparison, score=score, change=float(delta),
                             ci_lower=round(float(low), 9), ci_upper=round(float(high), 9),
                             seed=3324, draws=len(draws), examples_per_source=len(member)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lamini", "dolly"), default="lamini",
                        help="Downstream adaptation dataset (also selects default input/output paths)")
    parser.add_argument("--pythia-base", type=Path, default=PYTHIA_CACHE)
    parser.add_argument("--biogpt-base", type=Path, default=REPO / "artifacts/biogpt_large_mimir")
    parser.add_argument("--pythia-adapted", type=Path)
    parser.add_argument("--biogpt-adapted", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = REPO / "artifacts" / f"{args.dataset}_matched"
    results = REPO / "experiments/public_pythia_derivatives/results" / f"2026-11-24-{args.dataset}-matched"
    pythia_adapted = args.pythia_adapted or run / "pythia/mimir"
    biogpt_adapted = args.biogpt_adapted or run / "biogpt/mimir"
    audit_dir = args.audit_dir or run / "witness_audits"
    output_dir = args.output_dir or results
    label = {"lamini": "Lamini", "dolly": "Dolly"}[args.dataset]

    inputs = REPO / "data/processed/mimir_github"
    frozen = {split: load_rows(inputs / f"{split}.jsonl") for split in ("train", "test")}
    for split, count in (("train", 500), ("test", 200)):
        labels = [r["label"] for r in frozen[split]]
        if len(labels) != 2 * count or labels.count(1) != count or labels.count(0) != count:
            raise ValueError(f"Unexpected class counts in {split}")
    labels = np.array([r["label"] for r in frozen["test"]])
    n = int(sum(labels))
    individual_cutoff = math.sqrt(math.log(2 / .05) / n)
    two_target_cutoff = math.sqrt(math.log(4 / .05) / n)
    summary, decisions, lengths = [], {}, {}
    models = [("pythia_base", "Pythia-1.4B", args.pythia_base),
              ("pythia_adapted", f"Pythia-1.4B {label}", pythia_adapted),
              ("biogpt_base", "BioGPT-Large", args.biogpt_base),
              ("biogpt_adapted", f"BioGPT-Large {label}", biogpt_adapted)]
    for role, model_label, score_dir in models:
        tests, lengths[role] = verify_scores(score_dir, frozen)
        audit = fit_audit(score_dir, audit_dir / role, model_label)
        shuffled = {item["score_name"]: item for item in audit["shuffled_label_control"]}
        adapted = role.endswith("_adapted")
        for result in audit["main_results"]:
            score = result["score_name"]
            threshold = result["calibrated_threshold"]
            accepted = np.array([row[score] >= threshold for row in tests])
            decisions[role, score] = accepted
            metrics = result["test"]
            tp, fp = int(accepted[labels == 1].sum()), int(accepted[labels == 0].sum())
            if (tp, fp) != (metrics["tp"], metrics["fp"]):
                raise ValueError("Recovered acceptance counts disagree with repository audit")
            gap = (tp - fp) / n
            family = "descriptive"
            if adapted:
                family = "primary_MIN-K20" if score == SCORES[0] else "exploratory_mean"
            summary.append(dict(
                model=model_label, role=role, score=score, family=family, n_per_source=n,
                fitted_threshold=threshold, member_accepted=tp, nonmember_accepted=fp,
                signed_advantage=gap, absolute_advantage=abs(gap), auc=metrics["auc"],
                alpha=.05, individual_cutoff=individual_cutoff,
                positive_individual=gap > individual_cutoff,
                bonferroni_cutoff=two_target_cutoff if adapted else "",
                positive_bonferroni=(gap > two_target_cutoff) if adapted else "",
                shuffled_advantage=round(shuffled[score]["test_advantage"], 9),
            ))
    for family in ("pythia", "biogpt"):
        if lengths[f"{family}_adapted"] != lengths[f"{family}_base"]:
            raise ValueError(f"Base and adapted {family} tokenizer counts differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(output_dir / "witness_audit.csv", summary)
    write_table(output_dir / "witness_changes.csv", bootstrap_contrasts(decisions, labels))


if __name__ == "__main__":
    main()
