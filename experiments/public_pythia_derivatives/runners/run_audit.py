"""Run the submitted public-Pythia GitHub witness audit.

This runner covers all 14 public derivatives in main-paper Table 2 and
Appendix Table 6. It uses the paper's MIN-K 20% score plus a
pre-specified mean-log-probability secondary score.  For each seed it creates:

* a MIMIR GitHub member/nonmember calibration and held-out evaluation split;
* parent and verified-derived-target audits on that identical split;
* a shuffled-label calibration control for every main audit; and
* a nonmember-vs-nonmember control for every target audit.

Run on an allocated GPU node, for example:

    python experiments/public_pythia_derivatives/runners/run_audit.py \
      --seeds 0 --batch-size 4 --dtype float16

The companion compiler is:

    python experiments/public_pythia_derivatives/reports/compile_results.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
SCORES = "mean_logprob,min_k_20_logprob"
MIMIR_REVISION = "02500d3b7cece0cb7628e939ba9fc93fdb6362ae"
MODEL_REVISIONS = {
    "EleutherAI/pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "Leogrin/eleuther-pythia1b-hh-sft": "b29293529d7fb430257df6ae8079afeba2f42f9b",
    "EleutherAI/pythia-1.4b": "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
    "herMaster/pythia1.4B-finetuned-on-lamini-docs": "a6c699d6832bd3852c13d9c728c8599666b04c22",
    "kykim0/pythia-1.4b-tulu-v2-mix": "4d21350c5197fcfc6cac0b2db277c3e9a158b303",
    "LinguaCustodia/fin-pythia-1.4b": "818675a1325994d01daaec677b9ac40dddcc916d",
    "lomahony/pythia-1.4b-helpful-dpo": "599b0f9ff8bdbccb4ccc8bf8a6a3cb7770c1b800",
    "lomahony/pythia-1.4b-helpful-sft": "da0e31a28430fe2b03b009ebbfdbee74010931c2",
    "nnheui/pythia-1.4b-sft-full": None,
    "EleutherAI/pythia-6.9b": "c0e3eee36dc47af0c49f361c74cfe459c09f7f23",
    "allenai/open-instruct-pythia-6.9b-tulu": "958b73d2d0e92189bae7ef3bd4036f9b5b508bea",
    "lomahony/eleuther-pythia6.9b-hh-dpo": "f9eae2bc28c84768c79428f1affcff32e48cf688",
    "lomahony/eleuther-pythia6.9b-hh-sft": "8ca04ae6e59da4f9bd86ad8f3d760b99ac862b01",
    "pkarypis/pythia-ultrachat": "0561fb5d49b92fe0a3677f3f701503d55e8ed1af",
    "usvsnsp/pythia-6.9b-ppo": "9ee543790de178dbc99d5ef83ee51b0b4cd32441",
    "EleutherAI/pythia-12b": "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6",
    "lomahony/eleuther-pythia12b-hh-dpo": "6fc4cb69334241bbe93631f268c6f42bc05999d1",
    "lomahony/eleuther-pythia12b-hh-sft": "e12cdad97eb968c59d5d63de3e0057fac82ded88",
}


def pinned_revision(model: str) -> str:
    if model not in MODEL_REVISIONS:
        raise KeyError(f"No immutable revision declared for {model}")
    revision = MODEL_REVISIONS[model]
    if revision is None:
        raise RuntimeError(
            f"Cannot reproduce {model}: the original Hugging Face repository "
            "no longer serves its model files"
        )
    return revision


@dataclass(frozen=True)
class Family:
    name: str
    parent: str
    targets: tuple[str, ...]


# Exact target roster from submitted Appendix Table 6.
FAMILIES = {
    "1b": Family(
        name="Pythia-1B",
        parent="EleutherAI/pythia-1b",
        targets=("Leogrin/eleuther-pythia1b-hh-sft",),
    ),
    "1.4b": Family(
        name="Pythia-1.4B",
        parent="EleutherAI/pythia-1.4b",
        targets=(
            "herMaster/pythia1.4B-finetuned-on-lamini-docs",
            "kykim0/pythia-1.4b-tulu-v2-mix",
            "LinguaCustodia/fin-pythia-1.4b",
            "lomahony/pythia-1.4b-helpful-dpo",
            "lomahony/pythia-1.4b-helpful-sft",
            "nnheui/pythia-1.4b-sft-full",
        ),
    ),
    "6.9b": Family(
        name="Pythia-6.9B",
        parent="EleutherAI/pythia-6.9b",
        targets=(
            "allenai/open-instruct-pythia-6.9b-tulu",
            "lomahony/eleuther-pythia6.9b-hh-dpo",
            "lomahony/eleuther-pythia6.9b-hh-sft",
            "pkarypis/pythia-ultrachat",
            "usvsnsp/pythia-6.9b-ppo",
        ),
    ),
    "12b": Family(
        name="Pythia-12B",
        parent="EleutherAI/pythia-12b",
        targets=(
            "lomahony/eleuther-pythia12b-hh-dpo",
            "lomahony/eleuther-pythia12b-hh-sft",
        ),
    ),
}


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local vanilla Pythia GitHub provenance audits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help=f"Comma-separated subset of: {','.join(FAMILIES)}")
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated fixed MIMIR split seeds")
    parser.add_argument("--artifact-root",
                        default="artifacts/public_pythia_derivatives/paper_audit")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="Optional Hugging Face token; defaults to $HF_TOKEN")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="float16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--min-words", type=int, default=8)
    # MIMIR GitHub yields 963 usable member snippets after the required
    # 64-word preprocessing and deduplication.  560 + 400 therefore fits;
    # the submitted 560 + 400 split fits this source.
    parser.add_argument("--calibration-members", type=int, default=560)
    parser.add_argument("--calibration-nonmembers", type=int, default=340)
    parser.add_argument("--heldout-per-class", type=int, default=400)
    parser.add_argument("--resume", action="store_true",
                        help="Skip a step when its expected output already exists")
    parser.add_argument("--skip-nonmember-control", action="store_true",
                        help="Skip the nonmember-vs-nonmember target controls")

    parser.add_argument("--skip-unavailable", action="store_true",
                        help="Skip only roster entries with unavailable pinned weights")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare every requested seed, then exit before model scoring")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Require and reuse data prepared by the submission preflight")
    return parser.parse_args()


def run(command: list[str], expected: Path | None, *, resume: bool) -> None:
    if resume and expected and expected.exists():
        print(f"[resume] exists: {expected}", flush=True)
        return
    display = list(command)
    if "--token" in display:
        display[display.index("--token") + 1] = "<redacted>"
    print("\n+ " + " ".join(display), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode:
        raise SystemExit(f"{display[1]} failed with status {result.returncode}")


def prepare_data(args: argparse.Namespace, seed: int, seed_root: Path) -> tuple[Path, Path]:
    main_data = seed_root / "prepared" / "mimir_github"
    control_data = seed_root / "prepared" / "mimir_github_nonmember_control"
    token_args = ["--token", args.hf_token] if args.hf_token else []

    run([
        PYTHON, "scripts/data/prepare_mimir_domain.py",
        "--config", "github", "--split", "ngram_13_0.2",
        "--dataset-revision", MIMIR_REVISION,
        "--num-train-member", str(args.calibration_members),
        "--num-train-nonmember", str(args.calibration_nonmembers),
        "--num-test-per-class", str(args.heldout_per_class),
        "--max-words", str(args.max_words), "--min-words", str(args.min_words),
        "--seed", str(seed), "--output-dir", str(main_data), *token_args,
    ], main_data / "test.jsonl", resume=args.resume)

    if not args.skip_nonmember_control:
        run([
            PYTHON, "scripts/data/prepare_mimir_domain_nonmember_control.py",
            "--config", "github", "--ngram-split", "ngram_13_0.2",
            "--dataset-revision", MIMIR_REVISION,
            "--num-train-per-class", str(args.calibration_nonmembers),
            "--num-test-per-class", str(args.heldout_per_class),
            "--max-words", str(args.max_words), "--min-words", str(args.min_words),
            "--seed", str(seed), "--output-dir", str(control_data), *token_args,
        ], control_data / "test.jsonl", resume=args.resume)
    return main_data, control_data


def score_and_audit(
    args: argparse.Namespace,
    *,
    seed: int,
    seed_root: Path,
    model: str,
    main_data: Path,
    control_data: Path,
    parent_results: Path | None,
    is_target: bool,
) -> Path:
    model_slug = slug(model)
    main_scores = seed_root / "scores" / "main" / model_slug
    main_results = seed_root / "results" / "main" / model_slug

    run([
        PYTHON, "scripts/scoring/score_causal_lm_logprobs.py",
        "--model", model, "--model-revision", pinned_revision(model),
        "--train-file", str(main_data / "train.jsonl"),
        "--test-file", str(main_data / "test.jsonl"),
        "--output-dir", str(main_scores),
        "--min-k-pcts", "20", "--batch-size", str(args.batch_size),
        "--dtype", args.dtype, "--max-length", str(args.max_length),
    ], main_scores / "test_scores.jsonl", resume=args.resume)

    parent_arg = ["--parent-results", str(parent_results)] if parent_results else []
    run([
        PYTHON, "scripts/audit/run_threshold_audit.py",
        "--train-scores", str(main_scores / "train_scores.jsonl"),
        "--test-scores", str(main_scores / "test_scores.jsonl"),
        "--output-dir", str(main_results),
        "--score-keys", SCORES, "--primary-score", "min_k_20_logprob",
        "--model-label", model, "--run-shuffled-control", *parent_arg,
    ], main_results / "results.json", resume=args.resume)

    if is_target and not args.skip_nonmember_control:
        control_scores = seed_root / "scores" / "nonmember_control" / model_slug
        control_results = seed_root / "results" / "nonmember_control" / model_slug
        run([
            PYTHON, "scripts/scoring/score_causal_lm_logprobs.py",
            "--model", model, "--model-revision", pinned_revision(model),
            "--train-file", str(control_data / "train.jsonl"),
            "--test-file", str(control_data / "test.jsonl"),
            "--output-dir", str(control_scores),
            "--min-k-pcts", "20", "--batch-size", str(args.batch_size),
            "--dtype", args.dtype, "--max-length", str(args.max_length),
        ], control_scores / "test_scores.jsonl", resume=args.resume)
        run([
            PYTHON, "scripts/audit/run_threshold_audit.py",
            "--train-scores", str(control_scores / "train_scores.jsonl"),
            "--test-scores", str(control_scores / "test_scores.jsonl"),
            "--output-dir", str(control_results),
            "--score-keys", SCORES, "--primary-score", "min_k_20_logprob",
            "--model-label", model,
        ], control_results / "results.json", resume=args.resume)
    return main_results / "results.json"


def main() -> None:
    args = parse_args()
    if args.prepare_only and args.skip_prepare:
        raise SystemExit("--prepare-only and --skip-prepare are mutually exclusive")
    family_ids = [entry.strip() for entry in args.families.split(",") if entry.strip()]
    unknown = sorted(set(family_ids) - set(FAMILIES))
    if unknown:
        raise SystemExit(f"Unknown family id(s): {', '.join(unknown)}")
    seeds = [int(entry.strip()) for entry in args.seeds.split(",") if entry.strip()]
    artifact_root = (REPO_ROOT / args.artifact_root).resolve()

    print("Submitted public-Pythia GitHub witness audit")
    print(f"Families: {', '.join(family_ids)} | Seeds: {seeds}")
    print("Scores: MIN-K 20% (primary), mean log-probability (secondary)")
    print(f"Artifacts: {artifact_root}", flush=True)

    for seed in seeds:
        seed_root = artifact_root / f"seed_{seed}"
        if args.skip_prepare:
            main_data = seed_root / "prepared" / "mimir_github"
            control_data = seed_root / "prepared" / "mimir_github_nonmember_control"
            required = [main_data / "train.jsonl", main_data / "test.jsonl"]
            if not args.skip_nonmember_control:
                required.extend([control_data / "train.jsonl", control_data / "test.jsonl"])
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise SystemExit(
                    "Prepared data is missing; run the prepare job first:\n  "
                    + "\n  ".join(missing)
                )
        else:
            main_data, control_data = prepare_data(args, seed, seed_root)
        if args.prepare_only:
            continue
        for family_id in family_ids:
            family = FAMILIES[family_id]
            print(f"\n=== seed={seed}, family={family.name} ===", flush=True)
            parent_results = score_and_audit(
                args, seed=seed, seed_root=seed_root, model=family.parent,
                main_data=main_data, control_data=control_data,
                parent_results=None, is_target=False,
            )
            for target in family.targets:
                if args.skip_unavailable and MODEL_REVISIONS[target] is None:
                    print(f"Skipping unavailable submitted target: {target}", flush=True)
                    continue
                score_and_audit(
                    args, seed=seed, seed_root=seed_root, model=target,
                    main_data=main_data, control_data=control_data,
                    parent_results=parent_results, is_target=True,
                )

    if args.prepare_only:
        print("\nPrepared all requested seeds.")
    else:
        print("\nDone. Compile with:")
        print(f"  {PYTHON} experiments/public_pythia_derivatives/reports/compile_results.py --artifact-root {args.artifact_root}")


if __name__ == "__main__":
    main()
