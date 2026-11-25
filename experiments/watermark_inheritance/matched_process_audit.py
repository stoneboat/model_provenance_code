"""Verify frozen watermark scores and decisions for the three matched histories."""

import argparse
import csv
import math
from pathlib import Path

from .calibration_analysis import paired_cutoff
from .common import load_config, load_json, read_jsonl, write_json
from .score import calibrate_threshold, evaluate_shard


def verify_target(run, target, pair_ids):
    scores = run / "scores" / target
    examples = read_jsonl(scores / "examples.jsonl")
    frozen = load_json(scores / "thresholds.json")
    metrics = load_json(scores / "metrics.json")
    decisions = read_jsonl(scores / "audit_decisions.jsonl")
    if metrics["pilot"] or metrics["checkpoint"] != target:
        raise ValueError(f"Expected the final {target} target")
    if [(row["pair_id"], row["audit_split"]) for row in examples] != pair_ids:
        raise ValueError(f"Different prompt pairs or split for {target}")
    construction = [row for row in examples if row["audit_split"] == "calibration"]
    evaluation = [row for row in examples if row["audit_split"] == "audit"]
    if [r["pair_id"] for r in decisions] != [r["pair_id"] for r in evaluation]:
        raise ValueError(f"Different evaluation pairs for {target}")
    cutoff = paired_cutoff(0.05, len(evaluation))
    result = []
    for metric, field in [("watermark", "z"), ("min_k20", "min_k20")]:
        threshold = calibrate_threshold(*[[r[side][field] for r in construction]
                                          for side in ("s", "s_prime")])
        if frozen["detectors"][metric] != threshold:
            raise ValueError(f"Frozen {metric} threshold disagrees with the construction data")
        c = threshold["threshold"]
        decisions_at_c = [{side: row[side][field] > c for side in ("s", "s_prime")}
                          for row in evaluation]
        if [r[metric] for r in decisions] != decisions_at_c:
            raise ValueError(f"Frozen {metric} decisions disagree with the per-example scores")
        counts = [sum(row[side] for row in decisions_at_c) for side in ("s", "s_prime")]
        audit = evaluate_shard(*[[r[side][field] for r in evaluation] for side in ("s", "s_prime")], c)
        if {"threshold": c, **audit} != metrics["shard"][metric]:
            raise ValueError(f"Recomputed audit for {target} {metric} differs")
        gap = (counts[0] - counts[1]) / len(evaluation)
        assert math.isclose(gap, audit["signed_gap"], abs_tol=1e-14)
        result.append(dict(
            target=target, detector=metric, primary=metric=="watermark",
            ancestor_P=target=="A", exclusion_holds_by_design=target!="B",
            training_shard="S_double_prime" if target=="C" else "S",
            construction_pairs=len(construction), evaluation_pairs=len(evaluation),
            score_threshold=c, accepted_S=counts[0], accepted_S_prime=counts[1],
            signed_advantage=gap, absolute_advantage=abs(gap),
            alpha=0.05, paired_critical_value=cutoff, rejects=abs(gap)>cutoff,
            exact_swap_p=audit["p_shard"], positive_discordances=audit["positive_pairs"],
            negative_discordances=audit["negative_pairs"],
        ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.run_dir)
    pairs = read_jsonl(args.run_dir / "prepared/pairs.jsonl")
    pair_ids = [(row["pair_id"], row["audit_split"]) for row in pairs]
    assert len(set(row["pair_id"] for row in pairs)) == len(pair_ids) == config["pairs"] == 2000
    assert sum(split == "calibration" for _, split in pair_ids) == config["calibration_pairs"] == 400
    table = [row for target in ("A", "B", "C") for row in verify_target(args.run_dir, target, pair_ids)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(args.output_dir / "audit.json", {
        "decision_rule": "Strict absolute paired advantage above sqrt(2 log(2/alpha)/m).",
        "exclusion": "Conditional on the shared prompts, base, key, and fixed downstream data, every causal path from realized S to target weights passes through the designated parent P.",
        "primary_detector": "watermark", "diagnostic_detector": "min_k20",
        "comparison_family": "Per-target level 0.05; shared pairs; one realization per process.",
        "results": table,
    })


if __name__ == "__main__":
    main()
