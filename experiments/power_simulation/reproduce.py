"""Reproduce the three conditional-power rows reported in the oT8R response."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent


def rows(config):
    delta = config["conditional_advantage"]
    alpha = config["alpha"]
    trials = config["simulations"]
    for sample_size in config["sample_sizes"]:
        rng = np.random.default_rng(
            np.random.SeedSequence([config["seed"], config["case_id"], sample_size])
        )
        # The original simulator drew an activation indicator first. Every
        # replicate in this configuration is active.
        rng.random(trials)
        witness = rng.binomial(sample_size, np.full(trials, (1 + delta) / 2))
        control = rng.binomial(sample_size, np.full(trials, (1 - delta) / 2))
        cutoff = math.sqrt(math.log(2 / alpha) / sample_size)
        rejects = int((np.abs(witness - control) / sample_size > cutoff).sum())
        power_bound = max(0.0, (delta - cutoff) / (1 - cutoff))
        conditional_bound = -math.expm1(-sample_size * max(0.0, delta - cutoff) ** 2)
        yield {
            "sample_size": sample_size,
            "simulations": trials,
            "rejects": rejects,
            "empirical_power": rejects / trials,
            "proposition_6_power_lower": power_bound,
            "conditional_power_lower": conditional_bound,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with the archived result table.")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    result = list(rows(config))
    if args.check:
        with (HERE / "results.csv").open(newline="") as stream:
            expected = list(csv.DictReader(stream))
        if len(result) != len(expected) or any(
            not math.isclose(row[key], float(saved[key]), rel_tol=1e-12, abs_tol=1e-14)
            for row, saved in zip(result, expected) for key in row
        ):
            raise SystemExit("Reproduced power differs from the saved table.")
        print("Verified all three conditional-power rows.")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
