"""Recompute the reported witness-sharing audit from frozen numeric scores."""

import csv
import gzip
import math
from pathlib import Path

from experiments.watermark_inheritance.calibration_analysis import paired_cutoff
from experiments.watermark_inheritance.score import calibrate_threshold, evaluate_shard


RESULTS = Path(__file__).resolve().parents[1] / "results/m7vv-witness-sharing"


def verify():
    with gzip.open(RESULTS / "paired_scores.csv.gz", "rt", newline="") as stream:
        scores = list(csv.DictReader(stream))
    with (RESULTS / "audit.csv").open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(scores) != 6000 or len(expected) != 6:
        raise ValueError("Expected 2,000 paired scores per target and two detector results per target")
    for row in expected:
        metric = "z" if row["detector"] == "watermark" else "min_k20"
        pairs = [
            [[float(score[f"{side}_{metric}"]) for score in scores
              if score["split"] == split and score["target"] == row["target"]]
             for side in ("witness", "control")]
            for split in ("calibration", "audit")
        ]
        construction, (witness, control) = pairs
        if len(construction[0]) != 400 or len(witness) != 1600:
            raise ValueError(f"Unexpected number of pairs for {row['target']}")
        threshold = calibrate_threshold(*construction)["threshold"]
        audit = evaluate_shard(witness, control, threshold)
        counts = (sum(score > threshold for score in witness), sum(score > threshold for score in control))
        gap = (counts[0] - counts[1]) / len(witness)
        cutoff = paired_cutoff(float(row["alpha"]), len(witness))
        if (counts != (int(row["accepted_S"]), int(row["accepted_S_prime"]))
                or threshold != float(row["score_threshold"])
                or not math.isclose(gap, float(row["signed_advantage"]), abs_tol=1e-12)
                or not math.isclose(cutoff, float(row["paired_critical_value"]), abs_tol=1e-12)
                or (abs(gap) > cutoff) != (row["rejects"] == "True")
                or not math.isclose(audit["p_shard"], float(row["exact_swap_p"]), rel_tol=1e-12)):
            raise ValueError(f"Frozen audit differs for {row['target']} {row['detector']}")
    print("Verified three m7VV targets and both detectors from the paired scores.")


if __name__ == "__main__":
    verify()
