"""Release checks tying the public repository to the submitted paper tables."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_public_pythia_roster_matches_appendix_table_6() -> None:
    from experiments.public_pythia_derivatives.runners.run_audit import FAMILIES
    from src.shard_audit.experiments.experiment_registry import PYTHIA_TARGETS

    counts = {name: len(family.targets) for name, family in FAMILIES.items()}
    assert counts == {"1b": 1, "1.4b": 6, "6.9b": 5, "12b": 2}
    runner_targets = {
        target for family in FAMILIES.values() for target in family.targets
    }
    registry_targets = {target.hf_id for target in PYTHIA_TARGETS}
    submitted_targets = {
        row["target_model"]
        for row in read_rows(
            "experiments/public_pythia_derivatives/results/submitted_table_6.csv"
        )
    }
    assert runner_targets == registry_targets == submitted_targets
    assert "nnheui/pythia-1.4b-sft-full" in runner_targets


def test_submitted_llm_tables_have_the_reported_decisions() -> None:
    public_rows = read_rows(
        "experiments/public_pythia_derivatives/results/submitted_table_6.csv"
    )
    int8_rows = read_rows(
        "experiments/pythia_int8_quantization/results/submitted_table_7.csv"
    )
    assert len(public_rows) == 14
    assert len(int8_rows) == 6
    assert all(float(row["signed_shard_advantage"]) > float(row["threshold"]) for row in public_rows)
    assert all(float(row["signed_shard_advantage"]) > float(row["threshold"]) for row in int8_rows)
    assert Counter(row["decision"] for row in public_rows) == {"reject": 14}
    assert Counter(row["decision"] for row in int8_rows) == {"reject": 6}


def test_submitted_vit_tables_have_the_reported_controls() -> None:
    public_rows = read_rows(
        "experiments/vit_provenance/results/submitted_table_8.csv"
    )
    transformed_rows = read_rows(
        "experiments/vit_provenance/results/submitted_table_9.csv"
    )
    assert len(public_rows) == 8
    assert len(transformed_rows) == 12
    assert Counter(row["decision"] for row in public_rows) == {
        "reject": 7,
        "fail_to_reject": 1,
    }
    assert Counter(row["decision"] for row in transformed_rows) == {
        "reject": 11,
        "fail_to_reject": 1,
    }
    random_control = next(
        row for row in public_rows if "random initialization" in row["target_model"]
    )
    extreme_pruning = next(
        row for row in transformed_rows if row["transformation"].startswith("90%")
    )
    assert float(random_control["signed_shard_advantage"]) == pytest.approx(0.0062)
    assert float(extreme_pruning["signed_shard_advantage"]) == pytest.approx(0.0369)


def test_centered_logit_variance_uses_population_variance() -> None:
    torch = pytest.importorskip("torch")
    from src.shard_audit.scoring.vit_scores import centered_logit_variance

    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [1.0, 1.0, -0.5],
            [0.5, 1.5, 0.0],
            [2.5, -0.5, -1.0],
        ]
    )
    actual = centered_logit_variance(logits, batch_size=1, n_views=4)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    centered = log_probabilities - log_probabilities.mean(dim=-1, keepdim=True)
    expected = -centered.var(dim=0, unbiased=False).mean()
    assert actual.shape == (1,)
    assert actual.item() == pytest.approx(expected.item())

    # A class-specific offset shared by every view cannot change variance
    # across views after within-view centering.
    shifted = logits + torch.tensor([3.0, -2.0, 0.25])
    shifted_score = centered_logit_variance(shifted, batch_size=1, n_views=4)
    assert shifted_score.item() == pytest.approx(actual.item(), abs=1e-6)


def test_only_paper_and_rebuttal_experiment_packages_remain() -> None:
    names = {
        path.name
        for path in (ROOT / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert names == {
        "power_simulation",
        "public_pythia_derivatives",
        "pythia_int8_quantization",
        "vit_provenance",
        "watermark_inheritance",
    }
    assert not any(name.startswith("table_") for name in names)


def test_release_entrypoints_and_configs_exist() -> None:
    required = (
        "experiments/public_pythia_derivatives/configs/paper.yaml",
        "experiments/public_pythia_derivatives/jobs/prepare.sbatch",
        "experiments/public_pythia_derivatives/jobs/run_1b_1_4b.sbatch",
        "experiments/public_pythia_derivatives/jobs/run_6_9b.sbatch",
        "experiments/public_pythia_derivatives/jobs/run_12b.sbatch",
        "experiments/public_pythia_derivatives/jobs/compile.sbatch",
        "experiments/public_pythia_derivatives/submit.sh",
        "experiments/pythia_int8_quantization/configs/int8.yaml",
        "experiments/pythia_int8_quantization/jobs/run_paper_targets.sbatch",
        "experiments/vit_provenance/configs/clv.yaml",
        "experiments/vit_provenance/jobs/run_public_derivatives.sbatch",
        "experiments/watermark_inheritance/submit.sh",
        "experiments/watermark_inheritance/submit_ultrachat.sh",
    )
    assert all((ROOT / relative_path).is_file() for relative_path in required)


def test_threshold_audit_cli_runs_end_to_end(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    output = tmp_path / "result"
    for path, rows in (
        (
            train,
            [
                {"id": "tr-m0", "label": 1, "score": 1.0},
                {"id": "tr-m1", "label": 1, "score": 0.9},
                {"id": "tr-c0", "label": 0, "score": 0.1},
                {"id": "tr-c1", "label": 0, "score": 0.2},
            ],
        ),
        (
            test,
            [
                {"id": "te-m0", "label": 1, "score": 0.8},
                {"id": "te-m1", "label": 1, "score": 0.7},
                {"id": "te-c0", "label": 0, "score": 0.05},
                {"id": "te-c1", "label": 0, "score": 0.15},
            ],
        ),
    ):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    subprocess.run(
        [
            sys.executable,
            "scripts/audit/run_threshold_audit.py",
            "--train-scores",
            str(train),
            "--test-scores",
            str(test),
            "--output-dir",
            str(output),
            "--score-keys",
            "score",
            "--primary-score",
            "score",
            "--model-label",
            "synthetic-test",
            "--run-shuffled-control",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((output / "results.json").read_text())
    assert result["main_results"][0]["test"]["shard_advantage"] == 1.0
    assert result["shuffled_label_control"] is not None
    assert (output / "audit_report.md").is_file()
