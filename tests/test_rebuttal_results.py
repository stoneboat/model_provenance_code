"""Check the recorded rebuttal decisions and aggregate the retained calibration counts."""

from collections import defaultdict
from itertools import product
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.watermark_inheritance.calibration_analysis import paired_cutoff


ROOT = Path(__file__).resolve().parents[1]
WATERMARK = ROOT / "experiments/watermark_inheritance"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_witness_reuse_bonferroni_matches_the_submitted_trajectories():
    report = WATERMARK / "results/2026-11-21-reviewer-analysis"
    summary = json.loads((report / "submitted_48_multiplicity.json").read_text())
    for file, expected_hash in summary["inputs"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected_hash
    dolly = [
        ("Dolly", row["run"], row["checkpoint"], float(row["signed_gap"]))
        for row in rows(WATERMARK / "results/2026-10-06/main_results.csv")
        if row["checkpoint"] in ("T1", "T2", "T", "T0")
    ]
    ultra = [
        ("UltraChat", row["run_id"], row["checkpoint"], float(row["watermark_signed_gap"]))
        for row in rows(WATERMARK / "results/2026-10-06-distinct/curve.csv") if int(row["steps"]) > 0
    ]
    saved = rows(report / "submitted_48_multiplicity.csv")
    gaps = {(dataset, run, checkpoint): gap for dataset, run, checkpoint, gap in dolly + ultra}
    assert len(dolly) == 12
    assert len(ultra) == 36
    assert len(saved) == len(gaps) == 48
    adjusted = paired_cutoff(0.05 / len(saved), 8000)
    assert adjusted == pytest.approx(0.0434743615968706)
    assert sum(row["role"] == "target" for row in saved) == 27
    assert sum(row["role"] == "control" for row in saved) == 21
    for row in saved:
        key = (row["dataset"], row["run"], row["checkpoint"])
        gap = gaps[key]
        assert float(row["signed_gap"]) == gap
        assert (abs(gap) > adjusted) == (row["reject_bonferroni"] == "True") == (row["role"] == "target")
        assert (abs(gap) > paired_cutoff(0.05, 8000)) == (row["reject_unadjusted"] == "True")


def test_calibration_matches_the_reported_counts_and_grid():
    report = WATERMARK / "results/2026-11-23-reviewer-q4-calibration"
    config = json.loads((WATERMARK / "configs/aistats27_q4_calibration.json").read_text())
    endpoints = rows(report / "independent_unit_results.csv")
    assert len({row["model_seed"] for row in endpoints}) == len({row["shard_seed"] for row in endpoints}) == 15
    assert sum(row["dataset"] == "dolly" for row in endpoints) == 12
    assert sum(row["dataset"] == "ultrachat_controls" for row in endpoints) == 3
    cutoff = paired_cutoff(config["primary_alpha"], config["primary_sample_size"])
    for row in endpoints:
        gap = float(row["a"]) - float(row["a_prime"])
        assert gap == pytest.approx(float(row["signed_gap"]), abs=1e-12)
        assert abs(gap) < cutoff
        assert row["primary_false_alarm"] == "False"
    assert max(float(row["advantage"]) for row in endpoints) == pytest.approx(0.01575)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["primary_model_controls"]["one_sided_upper"] == pytest.approx(1 - 0.05 ** (1 / 15))

    counts = defaultdict(dict)
    for row in rows(report / "unit_clustered_sensitivity.csv"):
        key = (int(row["sample_size"]), float(row["alpha"]), float(row["contamination_rate"]))
        assert row["cluster_id"] not in counts[key]
        counts[key][row["cluster_id"]] = (int(row["subsample_replicates"]), int(row["primary_false_alarms"]))
    assert set(counts) == set(product(config["sample_sizes"], config["alpha_values"], config["contamination_rates"]))

    published = rows(report / "sensitivity_summary.csv")
    assert len(published) == len(counts)
    for row in published:
        key = (int(row["sample_size"]), float(row["alpha"]), float(row["contamination_rate"]))
        assert set(counts[key]) == {row["cluster_id"] for row in endpoints}
        total = sum(trials for trials, _ in counts[key].values())
        alarms = sum(rejections for _, rejections in counts[key].values())
        assert total == int(row["correlated_subsample_evaluations"])
        assert alarms == int(row["primary_false_alarms"])
        assert alarms / total == pytest.approx(float(row["primary_sensitivity_frequency"]))
        if key[0] == 250:
            assert alarms <= 1 and total == 375
        else:
            assert alarms == 0


def test_saved_alternative_scorer_and_deit_summaries():
    pythia = ROOT / "experiments/public_pythia_derivatives/results/2026-11-23-aistats27-q3"
    scorers = json.loads((pythia / "observed_scorer_summary.json").read_text())
    for scorer, detections in (("min_k_20_logprob", 13), ("mean_logprob", 12)):
        assert scorers[scorer] == {
            "available_targets_positive": detections,
            "available_targets_total": 13,
            "nonmember_control_false_alarms": 0,
        }
    vit = ROOT / "experiments/vit_provenance/results/2026-11-23-hard-negative/observed_summary.json"
    deit = json.loads(vit.read_text())
    threshold = math.sqrt(math.log(2 / 0.05) / deit["evaluation_pairs"])
    assert threshold == pytest.approx(deit["cutoff"])
    assert abs(deit["signed_shard_advantage"]) < threshold
    assert deit["decision"] == "fail_to_reject"
