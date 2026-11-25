"""Create immutable, seed-separated AISTATS-2027 control-run configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import load_json, write_json
from .configure import base_config


DEFAULT_STUDY = Path(__file__).parent / "configs/aistats27_false_positive.json"
SOURCE_FILES = (
    "common.py", "configure.py", "curve_prepare.py", "curve_train.py",
    "generate.py", "prepare.py", "reviewer_configure.py",
    "score.py", "train.py",
)


def zero_event_upper_bound(trials: int, confidence: float = 0.95) -> float:
    """One-sided exact upper bound after zero observed false alarms."""
    if trials < 1 or not 0 < confidence < 1:
        raise ValueError("Need positive trials and confidence strictly between zero and one")
    return 1 - (1 - confidence) ** (1 / trials)


def source_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def write_immutable_config(path: Path, config: dict) -> dict:
    """Publish once, tolerating a later commit of byte-identical source files."""
    if path.exists():
        existing = load_json(path)
        if existing == config:
            return existing
        # Upgrade only configs created by the pre-production local validation.
        candidate_without_digest = dict(config)
        candidate_without_digest.pop("reviewer_source_sha256")
        if "reviewer_source_sha256" not in existing and existing == candidate_without_digest:
            write_json(path, config)
            return config
        comparable_existing = dict(existing)
        comparable_candidate = dict(config)
        comparable_existing.pop("code_commit", None)
        comparable_candidate.pop("code_commit", None)
        if (
            existing.get("reviewer_source_sha256") == config["reviewer_source_sha256"]
            and comparable_existing == comparable_candidate
        ):
            return existing
        raise RuntimeError(f"Refusing to replace a different immutable config: {path}")
    write_json(path, config)
    return config


def configure(root: Path, shared_dir: Path, study_path: Path) -> dict:
    study = load_json(study_path)
    common = base_config(shared_dir)
    common["reviewer_source_sha256"] = source_digest()
    created = []
    groups = (("dolly", study["dolly_units"]), ("ultrachat_shards", study["ultrachat_units"]))
    all_model_seeds, all_shard_seeds = set(), set()
    for group, units in groups:
        for unit in units:
            if unit["model_seed"] in all_model_seeds or unit["shard_seed"] in all_shard_seeds:
                raise ValueError("Model and shard seeds must each be unique across independent units")
            all_model_seeds.add(unit["model_seed"])
            all_shard_seeds.add(unit["shard_seed"])
            run_dir = root / group / unit["unit"]
            config = dict(
                common,
                experiment="aistats27_false_positive_control",
                run_id=f"{group}:{unit['unit']}",
                # Keep the legacy field for compatibility, but generation and
                # training use the explicit seeds below.
                seed=unit["model_seed"],
                model_seed=unit["model_seed"],
                shard_seed=unit["shard_seed"],
                watermark_key=unit["watermark_key"],
                wrong_key=unit["wrong_key"],
                provenance_role="non_descendant_control",
                independent_unit=f"model:{unit['model_seed']}|shard:{unit['shard_seed']}",
                primary_checkpoint_name=("C3" if group == "dolly" else "control_step12500"),
            )
            path = run_dir / "config.json"
            config = write_immutable_config(path, config)
            write_immutable_config(run_dir / "config.lock.yaml", config)
            created.append(str(path.resolve()))
    pilot_source = load_json(root / "dolly" / study["dolly_units"][0]["unit"] / "config.json")
    pilot = dict(
        pilot_source,
        experiment="aistats27_false_positive_pilot",
        run_id="aistats27_false_positive_pilot",
        pairs=128,
        calibration_pairs=32,
        pilot_pairs=32,
        max_new_tokens=64,
        radio_token_budget=1000,
        pilot=True,
        log_every=1,
        primary_checkpoint_name="CP",
    )
    pilot_path = root / "pilot" / "config.json"
    pilot = write_immutable_config(pilot_path, pilot)
    write_immutable_config(root / "pilot" / "config.lock.yaml", pilot)
    created.append(str(pilot_path.resolve()))
    precision = study["precision_plan"]
    if abs(
        zero_event_upper_bound(precision["dolly_independent_units_after_completion"])
        - precision["dolly_zero_alarm_one_sided_95_percent_upper"]
    ) > 1e-6:
        raise ValueError("Dolly precision plan is inconsistent")
    manifest = {
        "study": study,
        "study_config": str(study_path.resolve()),
        "created_configs": created,
        "model_seeds": sorted(all_model_seeds),
        "shard_seeds": sorted(all_shard_seeds),
        "independence_policy": (
            "Only one-to-one model-seed/shard-seed units count as independent false-alarm trials; "
            "checkpoint repeats, subsamples, and cross-audits are correlated sensitivity analyses."
        ),
    }
    write_json(root / "study_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_inheritance/aistats27_reviewer"))
    parser.add_argument("--shared-dir", type=Path, default=Path(".cache/watermark"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args(argv)
    result = configure(args.root, args.shared_dir, args.study)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
