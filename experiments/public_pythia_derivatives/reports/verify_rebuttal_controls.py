"""Refit thresholds and verify the published frozen-split public controls."""

import csv
import gzip
import math
from pathlib import Path

from src.shard_audit.auditing.distinguishers import evaluate_at_threshold, select_threshold, sweep_thresholds


RESULTS = Path(__file__).resolve().parents[1] / "results/rebuttal-public-controls"


def verify():
    with gzip.open(RESULTS / "scores.csv.gz", "rt", newline="") as stream:
        scores = list(csv.DictReader(stream))
    with (RESULTS / "advantages.csv").open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(scores) != 8400 or len(expected) != 12:
        raise ValueError("Expected 1,400 scores per model and two detector results per model")
    for row in expected:
        partitions = []
        for split, count in (("train", 500), ("test", 200)):
            subset = [score for score in scores if score["model"] == row["model"] and score["split"] == split]
            labels = [int(score["label"]) for score in subset]
            if len(labels) != count * 2 or sum(labels) != count:
                raise ValueError(f"Unexpected {split} labels for {row['model']}")
            partitions.append((labels, [float(score[row["score"]]) for score in subset]))
        construction, evaluation = partitions
        threshold = select_threshold(sweep_thresholds(*construction, n_thresholds=2000))["threshold"]
        actual = evaluate_at_threshold(*evaluation, threshold)
        if (not math.isclose(threshold, float(row["threshold"]), abs_tol=1e-12)
                or actual["tp"] != int(row["member_accepted"])
                or actual["fp"] != int(row["nonmember_accepted"])
                or not math.isclose(actual["shard_advantage"], float(row["signed_advantage"]), abs_tol=1e-12)):
            raise ValueError(f"Frozen audit differs for {row['model']} {row['score']}")
    print("Verified six public checkpoints and both scores from the frozen numeric data.")


if __name__ == "__main__":
    verify()
