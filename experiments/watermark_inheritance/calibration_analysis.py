"""Cheap Reviewer 3 Q4 calibration from frozen per-example score caches.

This module never loads a model.  It preserves each construction-fitted score
threshold, subsamples complete held-out prompt pairs, and recomputes only the
prespecified paired hypothesis-test critical value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta

from .common import load_json, read_jsonl, write_json
from .score import paired_label_swap


DEFAULT_CONFIG = Path(__file__).parent / "configs/aistats27_q4_calibration.json"


def paired_cutoff(alpha: float, sample_size: int) -> float:
    """Conservative paired Hoeffding cutoff used by the manuscript."""
    if not 0 < alpha < 1 or sample_size < 1:
        raise ValueError("alpha must be in (0, 1) and sample_size must be positive")
    return math.sqrt(2 * math.log(2 / alpha) / sample_size)


def clopper_pearson(events: int, trials: int, confidence: float = 0.95) -> dict:
    """Exact two-sided interval and one-sided upper confidence bound."""
    if not 0 <= events <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid binomial confidence-interval inputs")
    tail = (1 - confidence) / 2
    lower = 0.0 if events == 0 else float(beta.ppf(tail, events, trials - events + 1))
    upper = 1.0 if events == trials else float(beta.ppf(1 - tail, events + 1, trials - events))
    one_sided_upper = (
        1.0 if events == trials else float(beta.ppf(confidence, events + 1, trials - events))
    )
    return {
        "events": events,
        "trials": trials,
        "frequency": events / trials,
        "confidence": confidence,
        "two_sided_lower": lower,
        "two_sided_upper": upper,
        "one_sided_upper": one_sided_upper,
    }


def holm_rejections(p_values: list[float], alpha: float) -> list[bool]:
    """Holm step-down decisions, valid without independence assumptions."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return rejected


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _unit_metadata(run_dir: Path, family: str, dataset: str, checkpoint: str) -> dict:
    config = load_json(run_dir / "config.json")
    return {
        "family": family,
        "dataset": dataset,
        "run_id": config["run_id"],
        "model_seed": int(config["model_seed"]),
        "shard_seed": int(config["shard_seed"]),
        "checkpoint": checkpoint,
        "cluster_id": f"model:{config['model_seed']}|shard:{config['shard_seed']}",
        "run_dir": str(run_dir),
    }


def load_model_unit(run_dir: Path, dataset: str, checkpoint: str) -> dict:
    score_dir = run_dir / "scores" / checkpoint
    threshold = float(load_json(score_dir / "thresholds.json")["detectors"]["watermark"]["threshold"])
    examples = [row for row in read_jsonl(score_dir / "examples.jsonl") if row["audit_split"] == "audit"]
    if len(examples) != 8000 or len({row["pair_id"] for row in examples}) != len(examples):
        raise ValueError(f"Expected 8,000 unique held-out pairs: {score_dir}")
    unit = _unit_metadata(run_dir, "independent_model_control", dataset, checkpoint)
    unit.update({
        "threshold": threshold,
        "pair_ids": [row["pair_id"] for row in examples],
        "first": [row["s"]["z"] > threshold for row in examples],
        "second": [row["s_prime"]["z"] > threshold for row in examples],
        "input_paths": [str(score_dir / "examples.jsonl"), str(score_dir / "thresholds.json")],
    })
    return unit


def evaluate_decisions(first: list[bool], second: list[bool], alpha: float) -> dict:
    if not first or len(first) != len(second):
        raise ValueError("Paired decisions must be nonempty and aligned")
    sample_size = len(first)
    signed_gap = sum(int(left) - int(right) for left, right in zip(first, second)) / sample_size
    exact = paired_label_swap(first, second)
    cutoff = paired_cutoff(alpha, sample_size)
    hoeffding_p_bound = min(1.0, 2 * math.exp(-sample_size * signed_gap**2 / 2))
    return {
        "sample_size": sample_size,
        "alpha": alpha,
        "a": sum(first) / sample_size,
        "a_prime": sum(second) / sample_size,
        "signed_gap": signed_gap,
        "advantage": abs(signed_gap),
        "paired_cutoff": cutoff,
        "primary_false_alarm": abs(signed_gap) > cutoff,
        "hoeffding_p_bound": hoeffding_p_bound,
        "exact_p_value": exact["p_shard"],
        "exact_sign_rejection": exact["p_shard"] <= alpha,
        "discordant_pairs": exact["discordant_pairs"],
        "positive_pairs": exact["positive_pairs"],
        "negative_pairs": exact["negative_pairs"],
    }


def discover_primary_units(root: Path, config: dict) -> list[dict]:
    units = []
    for dataset, checkpoint in config["primary_model_endpoints"].items():
        pattern = "unit_*" if dataset == "dolly" else "run_*"
        for run_dir in sorted((root / dataset).glob(pattern)):
            units.append(load_model_unit(run_dir, dataset, checkpoint))
    if len(units) != 15 or len({unit["cluster_id"] for unit in units}) != 15:
        raise ValueError("Expected 15 distinct model/shard endpoints")
    return units


def primary_table(units: list[dict], alpha: float) -> list[dict]:
    rows = []
    for unit in units:
        result = evaluate_decisions(unit["first"], unit["second"], alpha)
        rows.append({key: unit[key] for key in (
            "family", "dataset", "run_id", "model_seed", "shard_seed", "checkpoint", "cluster_id"
        )} | result)
    p_values = [row["exact_p_value"] for row in rows]
    exact_holm = holm_rejections(p_values, alpha)
    primary_holm = holm_rejections([row["hoeffding_p_bound"] for row in rows], alpha)
    family_size = len(rows)
    adjusted_cutoff = paired_cutoff(alpha / family_size, len(units[0]["first"]))
    for row, exact_holm_reject, primary_holm_reject in zip(rows, exact_holm, primary_holm):
        row["family_size"] = family_size
        row["bonferroni_alpha"] = alpha / family_size
        row["bonferroni_paired_cutoff"] = adjusted_cutoff
        row["bonferroni_primary_false_alarm"] = row["advantage"] > adjusted_cutoff
        row["holm_primary_false_alarm"] = primary_holm_reject
        row["bonferroni_exact_rejection"] = row["exact_p_value"] <= alpha / family_size
        row["holm_exact_rejection"] = exact_holm_reject
    return rows


def _subsample_indices(unit: dict, seed: int, replicate: int) -> list[int]:
    indices = list(range(len(unit["pair_ids"])))
    random.Random(_stable_seed(seed, unit["family"], unit["run_id"], replicate, "pairs")).shuffle(indices)
    return indices


def _contaminate(first: list[bool], second: list[bool], rate: float, seed: int,
                 unit: dict, replicate: int, sample_size: int) -> list[bool]:
    if not 0 <= rate < 1:
        raise ValueError("Contamination rate must be in [0, 1)")
    count = math.floor(rate * sample_size)
    order = list(range(sample_size))
    random.Random(_stable_seed(
        seed, unit["family"], unit["run_id"], replicate, sample_size, "contamination"
    )).shuffle(order)
    contaminated = list(second)
    for index in order[:count]:
        contaminated[index] = first[index]
    return contaminated


def sensitivity_tables(units: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    unit_rows = []
    seed = int(config["analysis_seed"])
    repetitions = int(config["subsample_replicates"])
    for unit in units:
        for sample_size in config["sample_sizes"]:
            effective_repetitions = 1 if sample_size == len(unit["first"]) else repetitions
            counters = defaultdict(lambda: {"primary": 0, "exact": 0, "advantages": []})
            for replicate in range(effective_repetitions):
                selected = _subsample_indices(unit, seed, replicate)[:sample_size]
                first = [unit["first"][index] for index in selected]
                second = [unit["second"][index] for index in selected]
                for rate in config["contamination_rates"]:
                    altered = _contaminate(first, second, rate, seed, unit, replicate, sample_size)
                    for alpha in config["alpha_values"]:
                        result = evaluate_decisions(first, altered, alpha)
                        key = (float(rate), float(alpha))
                        counters[key]["primary"] += int(result["primary_false_alarm"])
                        counters[key]["exact"] += int(result["exact_sign_rejection"])
                        counters[key]["advantages"].append(result["advantage"])
            for (rate, alpha), values in sorted(counters.items()):
                unit_rows.append({
                    **{key: unit[key] for key in (
                        "family", "dataset", "run_id", "model_seed", "shard_seed", "checkpoint", "cluster_id"
                    )},
                    "sample_size": sample_size,
                    "alpha": alpha,
                    "contamination_rate": rate,
                    "subsample_replicates": effective_repetitions,
                    "paired_cutoff": paired_cutoff(alpha, sample_size),
                    "primary_false_alarms": values["primary"],
                    "primary_false_alarm_frequency": values["primary"] / effective_repetitions,
                    "exact_sign_rejections": values["exact"],
                    "exact_sign_rejection_frequency": values["exact"] / effective_repetitions,
                    "mean_advantage": sum(values["advantages"]) / effective_repetitions,
                    "max_advantage": max(values["advantages"]),
                })
    grouped = defaultdict(list)
    for row in unit_rows:
        grouped[(row["family"], row["sample_size"], row["alpha"], row["contamination_rate"])].append(row)
    summary_rows = []
    for (family, sample_size, alpha, rate), rows in sorted(grouped.items()):
        total_repetitions = sum(row["subsample_replicates"] for row in rows)
        primary = sum(row["primary_false_alarms"] for row in rows)
        exact = sum(row["exact_sign_rejections"] for row in rows)
        summary_rows.append({
            "family": family,
            "sample_size": sample_size,
            "alpha": alpha,
            "contamination_rate": rate,
            "independent_unit_clusters": len(rows),
            "correlated_subsample_evaluations": total_repetitions,
            "primary_false_alarms": primary,
            "primary_sensitivity_frequency": primary / total_repetitions,
            "exact_sign_rejections": exact,
            "exact_sign_sensitivity_frequency": exact / total_repetitions,
            "analysis_unit": "model/shard cluster; repeated subsets are sensitivity analyses, not independent trials",
        })
    return unit_rows, summary_rows


def _input_manifest(units: list[dict]) -> list[dict]:
    records = []
    seen = set()
    for unit in units:
        for name in unit["input_paths"]:
            if name in seen:
                continue
            seen.add(name)
            path = Path(name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
    return records


def run(root: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> dict:
    config = load_json(config_path)
    units = discover_primary_units(root, config)
    rows = primary_table(units, float(config["primary_alpha"]))
    unit_sensitivity, sensitivity_summary = sensitivity_tables(units, config)
    _write_csv(output_dir / "independent_unit_results.csv", rows)
    _write_csv(output_dir / "unit_clustered_sensitivity.csv", unit_sensitivity)
    _write_csv(output_dir / "sensitivity_summary.csv", sensitivity_summary)
    summary = {
        "primary_model_controls": clopper_pearson(sum(row["primary_false_alarm"] for row in rows), len(rows)),
        "input_manifest": _input_manifest(units),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_inheritance/aistats27_reviewer"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/watermark_inheritance/aistats27_calibration_report"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
