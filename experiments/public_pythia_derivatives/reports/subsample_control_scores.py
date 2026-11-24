"""Compare two controls using the same held-out MIMIR rows in each subsample."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom


SCORES = ("min_k_20_logprob", "mean_logprob")
SIZES = (25, 50, 100, 150, 200)
ALPHAS = (0.01, 0.025, 0.05, 0.10)
DRAWS = 10000
SEED = 3324


def read_control(directory):
    audit = json.loads((directory / "results.json").read_text())["main_results"]
    rules = {r["score_name"]: r for r in audit}
    with (directory / "test_scores.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(r["label"], r["id"], r.get("text_hash")) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate held-out keys in {directory}")
    decisions = []
    for score in SCORES:
        threshold = rules[score]["calibrated_threshold"]
        values = np.array([r[score] for r in rows])
        if not np.all(np.isfinite(values)) or not math.isfinite(threshold):
            raise ValueError(f"Non-finite evaluation score or fitted threshold for {score}")
        accepted = values >= threshold
        metric = rules[score]["test"]
        for label, count in ((1, "tp"), (0, "fp")):
            selected = [i for i, k in enumerate(keys) if k[0] == label]
            if accepted[selected].sum() != metric[count]:
                raise ValueError(f"Held-out decisions disagree with saved {score} audit")
        decisions.append(accepted)
    return keys, np.column_stack(decisions)


def finite_pool_probability(pool_size, m, members_accepted, controls_accepted, cutoff):
    counts = np.arange(m + 1)
    members = hypergeom.pmf(counts, pool_size, members_accepted, m)
    controls = hypergeom.pmf(counts, pool_size, controls_accepted, m)
    crosses = np.abs(counts[:, None] - counts) / m > cutoff
    return np.outer(members, controls)[crosses].sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biogpt-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keys, biogpt = read_control(args.biogpt_dir)
    random_keys, random = read_control(args.random_dir)
    if keys != random_keys:
        raise ValueError("The two models must have the same held-out rows in the same order")
    labels = np.array([key[0] for key in keys])
    decisions = np.column_stack([biogpt, random])
    members, controls = decisions[labels == 1], decisions[labels == 0]
    pool_size = len(members)
    if pool_size != 200 or len(controls) != pool_size or len(keys) != 2 * pool_size:
        raise ValueError("Expected 200 held-out rows per source")
    last_rows = np.array(SIZES) - 1
    rng = np.random.default_rng(SEED)
    gaps = np.empty((DRAWS, len(SIZES), decisions.shape[1]), dtype=float)
    for draw in range(DRAWS):
        member_counts = members[rng.permutation(pool_size)].cumsum(axis=0)[last_rows]
        control_counts = controls[rng.permutation(pool_size)].cumsum(axis=0)[last_rows]
        gaps[draw] = (member_counts - control_counts) / np.array(SIZES)[:, None]

    model_scores = [(name, score) for name in ("BioGPT-Large", "Pythia-1.4B step0")
                    for score in SCORES]
    output = []
    for col, (model, score) in enumerate(model_scores):
        accepted_member = int(members[:, col].sum())
        accepted_control = int(controls[:, col].sum())
        for j, m in enumerate(SIZES):
            for alpha in ALPHAS:
                cutoff = math.sqrt(math.log(2 / alpha) / m)
                crossings = int(np.count_nonzero(np.abs(gaps[:, j, col]) > cutoff))
                exact = finite_pool_probability(pool_size, m, accepted_member, accepted_control, cutoff)
                output.append(dict(model=model, score=score, alpha=alpha, sample_per_source=m,
                                   seed=SEED, subsamples=DRAWS, rejections=crossings,
                                   observed_frequency=crossings/DRAWS, exact_probability=exact,
                                   cutoff=cutoff, pool_per_source=pool_size,
                                   pool_members_accepted=accepted_member,
                                   pool_controls_accepted=accepted_control,
                                   pool_signed_advantage=(accepted_member-accepted_control)/pool_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
