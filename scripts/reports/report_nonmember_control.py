"""Generate nonmember-vs-nonmember control summary report.

Reads parent and target threshold-experiment results from the control run
(where both pseudo-classes are MIMIR GitHub nonmembers) and produces a
Markdown summary comparing control advantages to the main experiment's
member-vs-nonmember advantages.

Usage:
    python scripts/reports/report_nonmember_control.py \\
        --parent-results outputs/runs/nonmember_control_seed0_pythia_1_4b/results.json \\
        --target-results outputs/runs/nonmember_control_seed0_nnheui_pythia_1_4b_sft_full/results.json \\
        --data-manifest data/processed/mimir_github_nonmember_control_seed0/manifest.json \\
        --output-dir outputs/reports/nonmember_vs_nonmember_control_parent_target
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Main experiment reference values
MAIN_PARENT_ADV = {
    "mean_logprob":     0.400,
    "min_k_5_logprob":  0.275,
    "min_k_10_logprob": 0.315,
    "min_k_20_logprob": 0.380,
    "min_k_40_logprob": 0.385,
}
MAIN_TARGET_ADV = {
    "mean_logprob":     0.350,
    "min_k_5_logprob":  0.290,
    "min_k_10_logprob": 0.335,
    "min_k_20_logprob": 0.375,
    "min_k_40_logprob": 0.370,
}
MAIN_PARENT_AUC = {
    "mean_logprob":     0.754,
    "min_k_5_logprob":  0.689,
    "min_k_10_logprob": 0.708,
    "min_k_20_logprob": 0.736,
    "min_k_40_logprob": 0.750,
}
MAIN_TARGET_AUC = {
    "mean_logprob":     0.747,
    "min_k_5_logprob":  0.716,
    "min_k_10_logprob": 0.727,
    "min_k_20_logprob": 0.741,
    "min_k_40_logprob": 0.748,
}
SCORE_KEYS = [
    "mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
    "min_k_20_logprob", "min_k_40_logprob",
]
PRIMARY = "min_k_20_logprob"


def _load_results(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "main_results" in raw:
        lst = raw["main_results"]
    elif isinstance(raw, list):
        lst = raw
    else:
        raise ValueError(f"Unrecognized results format in {path}")
    return {r["score_name"]: r for r in lst}


def _fmt(v, decimals=3):
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def _null_interp(adv: float, null_bound: float) -> str:
    if adv <= 0.10:
        return "good null behavior (≤0.10)"
    if adv <= null_bound:
        return f"within finite-sample null bound ({null_bound:.3f})"
    if adv <= 0.20:
        return "moderate — possible finite-sample noise"
    return "HIGH — suspicious; investigate confounds"


def _write_md(lines: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Report written to %s", path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate nonmember-vs-nonmember control summary report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parent-results", required=True,
                   help="Path to parent results.json from the control run")
    p.add_argument("--target-results", required=True,
                   help="Path to target results.json from the control run")
    p.add_argument("--data-manifest", required=True,
                   help="Path to data manifest.json from control data preparation")
    p.add_argument("--output-dir",
                   default="outputs/reports/nonmember_vs_nonmember_control_parent_target")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading parent results:  %s", args.parent_results)
    logger.info("Loading target results:  %s", args.target_results)
    logger.info("Loading data manifest:   %s", args.data_manifest)

    parent_res = _load_results(args.parent_results)
    target_res = _load_results(args.target_results)

    with open(args.data_manifest) as f:
        manifest = json.load(f)

    n_test = manifest.get("n_test_per_class", 200)
    null_bound = manifest.get("null_bound_alpha05") or math.sqrt(math.log(40) / n_test)
    seed = manifest.get("seed", 0)
    n_train = manifest.get("n_train_per_class", "?")

    p_primary = parent_res.get(PRIMARY, {})
    t_primary = target_res.get(PRIMARY, {})
    p_ctrl_adv = p_primary.get("test", {}).get("shard_advantage", float("nan"))
    t_ctrl_adv = t_primary.get("test", {}).get("shard_advantage", float("nan"))

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    logger.info("\n=== Key Control Results ===")
    logger.info("%-30s %8s %8s %8s", "Score", "P_Ctrl", "T_Ctrl", "P_Main")
    for key in SCORE_KEYS:
        p_adv = parent_res.get(key, {}).get("test", {}).get("shard_advantage", None)
        t_adv = target_res.get(key, {}).get("test", {}).get("shard_advantage", None)
        m_adv = MAIN_PARENT_ADV.get(key)
        logger.info("%-30s %8s %8s %8s", key,
                    _fmt(p_adv), _fmt(t_adv), _fmt(m_adv))

    # ------------------------------------------------------------------ #
    # Build Markdown report
    # ------------------------------------------------------------------ #
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        "# Nonmember-vs-Nonmember Control: Parent and Target",
        "",
        f"**Generated:** {now_utc}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Goal | Verify distinguisher shows near-zero advantage when both pseudo-shards are nonmembers |",
        "| Parent model | `EleutherAI/pythia-1.4b` |",
        "| Target model | `nnheui/pythia-1.4b-sft-full` |",
        "| Dataset | MIMIR GitHub (`iamgroot42/mimir`, config=`github`) |",
        f"| Source pool | MIMIR GitHub nonmember/test side (split=`{manifest.get('ngram_split','ngram_13_0.2')}`) |",
        f"| Pseudo-shard sizes | {n_train} train + {n_test} test per pseudo-class |",
        f"| Seed | {seed} |",
        f"| Primary score | `{PRIMARY}` |",
        f"| Parent control advantage | **{_fmt(p_ctrl_adv)}** |",
        f"| Target control advantage | **{_fmt(t_ctrl_adv)}** |",
        f"| Finite-sample null bound (α=0.05) | {_fmt(null_bound)} |",
        "",
        "**Main conclusion:**",
        "",
    ]

    # Interpret parent
    p_interp = _null_interp(abs(p_ctrl_adv) if p_ctrl_adv == p_ctrl_adv else 1.0, null_bound)
    t_interp = _null_interp(abs(t_ctrl_adv) if t_ctrl_adv == t_ctrl_adv else 1.0, null_bound)

    lines += [
        f"Parent control advantage: **{_fmt(p_ctrl_adv)}** — {p_interp}.",
        f"Target control advantage: **{_fmt(t_ctrl_adv)}** — {t_interp}.",
        "",
        "Main member-vs-nonmember advantages: parent **0.380**, target **0.375** (`min_k_20_logprob`).",
        "",
        "---",
        "",
        "## 2. Data Construction",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| MIMIR config | `{manifest.get('config','github')}` |",
        f"| MIMIR n-gram split | `{manifest.get('ngram_split','ngram_13_0.2')}` |",
        f"| Source file | `{manifest.get('source_file','?')}` |",
        "| Both classes true membership | **nonmember** (w.r.t. Pythia pretraining) |",
        "| Pseudo-label definitions | `label=1` = `nonmember_a` (S0); `label=0` = `nonmember_b` (S1) |",
        f"| Train per pseudo-class | {n_train} |",
        f"| Test per pseudo-class | {n_test} |",
        f"| Seed | {seed} |",
        "| Preprocessing | normalize whitespace; min 8 words; truncate to 32 words; no chat template |",
        f"| Raw nonmember pool | {manifest.get('n_raw','?')} texts |",
        f"| After preprocessing | {manifest.get('n_after_preprocessing','?')} texts |",
        f"| S0 / S1 hash overlap | {manifest.get('s0_s1_hash_overlap', 0)} |",
        f"| Train / test hash overlap | {manifest.get('train_test_hash_overlap', 0)} |",
        "",
        "> **Label warning**: These are artificial control labels.",
        "> They do NOT indicate member vs nonmember.",
        "> Both classes are nonmembers with respect to the Pythia pretraining membership label.",
        "",
        "---",
        "",
        "## 3. Parent Control Results",
        "",
        "| Score | Test Acc | Bal Acc | AUC | TPR@1%FPR | Shard Adv | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key in SCORE_KEYS:
        r = parent_res.get(key, {})
        test = r.get("test", {})
        lines.append(
            f"| `{key}` "
            f"| {_fmt(test.get('accuracy'))} "
            f"| {_fmt(test.get('balanced_accuracy'))} "
            f"| {_fmt(test.get('auc'))} "
            f"| {_fmt(test.get('tpr_at_1_fpr'))} "
            f"| {_fmt(test.get('shard_advantage'))} "
            f"| {_fmt(r.get('calibrated_threshold'), 4)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Target Control Results",
        "",
        "| Score | Test Acc | Bal Acc | AUC | TPR@1%FPR | Shard Adv | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key in SCORE_KEYS:
        r = target_res.get(key, {})
        test = r.get("test", {})
        lines.append(
            f"| `{key}` "
            f"| {_fmt(test.get('accuracy'))} "
            f"| {_fmt(test.get('balanced_accuracy'))} "
            f"| {_fmt(test.get('auc'))} "
            f"| {_fmt(test.get('tpr_at_1_fpr'))} "
            f"| {_fmt(test.get('shard_advantage'))} "
            f"| {_fmt(r.get('calibrated_threshold'), 4)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Comparison to Main Member-vs-Nonmember Experiment",
        "",
        "| Model | Main Adv (min_k_20) | Control Adv (min_k_20) | Difference |",
        "|---|---:|---:|---:|",
    ]

    p_diff = p_ctrl_adv - 0.380 if p_ctrl_adv == p_ctrl_adv else float("nan")
    t_diff = t_ctrl_adv - 0.375 if t_ctrl_adv == t_ctrl_adv else float("nan")

    def _fmt_signed(v):
        if v != v:  # nan
            return "n/a"
        return f"{v:+.3f}"

    lines += [
        f"| Parent | 0.380 | {_fmt(p_ctrl_adv)} | {_fmt_signed(p_diff)} |",
        f"| Target | 0.375 | {_fmt(t_ctrl_adv)} | {_fmt_signed(t_diff)} |",
        "",
        "Full score comparison:",
        "",
        "| Score | Main P Adv | Ctrl P Adv | Main T Adv | Ctrl T Adv |",
        "|---|---:|---:|---:|---:|",
    ]

    for key in SCORE_KEYS:
        p_main = MAIN_PARENT_ADV.get(key, None)
        t_main = MAIN_TARGET_ADV.get(key, None)
        p_ctrl = parent_res.get(key, {}).get("test", {}).get("shard_advantage", None)
        t_ctrl = target_res.get(key, {}).get("test", {}).get("shard_advantage", None)
        lines.append(
            f"| `{key}` "
            f"| {_fmt(p_main)} "
            f"| {_fmt(p_ctrl)} "
            f"| {_fmt(t_main)} "
            f"| {_fmt(t_ctrl)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Null Interpretation",
        "",
        f"**Finite-sample null bound** (α=0.05, m={n_test} per class):",
        "",
        f"  √(log(2/0.05) / {n_test}) = √(log(40) / {n_test}) ≈ {null_bound:.3f}",
        "",
        "This is the expected scale of random fluctuation in advantage when the null",
        "hypothesis (no real distinction between shards) is true.",
        "",
        "**Interpretation thresholds:**",
        "",
        "| Range | Interpretation |",
        "|---|---|",
        "| Adv ≤ 0.10 | Good null behavior |",
        f"| 0.10 < Adv ≤ {null_bound:.2f} | Within finite-sample null bound |",
        "| 0.10 < Adv ≤ 0.20 | Moderate — possible finite-sample or threshold-selection noise |",
        "| Adv > 0.20 | Suspicious — investigate confounds |",
        "",
        "**Parent control:**",
        "",
        f"- `{PRIMARY}` advantage: **{_fmt(p_ctrl_adv)}** — {p_interp}.",
        "",
        "**Target control:**",
        "",
        f"- `{PRIMARY}` advantage: **{_fmt(t_ctrl_adv)}** — {t_interp}.",
        "",
        "---",
        "",
        "## 7. Commands Run",
        "",
        "```bash",
        "# 1. Prepare control data",
        "python scripts/data/prepare_mimir_github_nonmember_control.py \\",
        f"  --config {manifest.get('config','github')} \\",
        f"  --ngram-split {manifest.get('ngram_split','ngram_13_0.2')} \\",
        f"  --num-train-per-class {n_train} \\",
        f"  --num-test-per-class {n_test} \\",
        f"  --seed {seed} \\",
        f"  --output-dir {manifest.get('output_dir','data/processed/mimir_github_nonmember_control_seed0')}",
        "",
        "# 2. Score parent on control data",
        "python scripts/scoring/extract_logprob_scores.py \\",
        "  --model EleutherAI/pythia-1.4b \\",
        f"  --train-file data/processed/mimir_github_nonmember_control_seed{seed}/train.jsonl \\",
        f"  --test-file  data/processed/mimir_github_nonmember_control_seed{seed}/test.jsonl \\",
        f"  --output-dir data/scores/mimir_github_nonmember_control_seed{seed}_pythia_1_4b \\",
        "  --min-k-pcts 5,10,20,40 --batch-size 4",
        "",
        "# 3. Run parent threshold experiment",
        "python scripts/experiments/run_mia_experiment.py \\",
        f"  --train-scores data/scores/mimir_github_nonmember_control_seed{seed}_pythia_1_4b/train_scores.jsonl \\",
        f"  --test-scores  data/scores/mimir_github_nonmember_control_seed{seed}_pythia_1_4b/test_scores.jsonl \\",
        f"  --output-dir   outputs/runs/nonmember_control_seed{seed}_pythia_1_4b \\",
        "  --primary-score min_k_20_logprob",
        "",
        "# 4. Score target on control data",
        "python scripts/scoring/extract_logprob_scores.py \\",
        "  --model nnheui/pythia-1.4b-sft-full \\",
        f"  --train-file data/processed/mimir_github_nonmember_control_seed{seed}/train.jsonl \\",
        f"  --test-file  data/processed/mimir_github_nonmember_control_seed{seed}/test.jsonl \\",
        f"  --output-dir data/scores/mimir_github_nonmember_control_seed{seed}_nnheui_pythia_1_4b_sft_full \\",
        "  --min-k-pcts 5,10,20,40 --batch-size 4",
        "",
        "# 5. Run target threshold experiment",
        "python scripts/experiments/run_mia_experiment.py \\",
        f"  --train-scores data/scores/mimir_github_nonmember_control_seed{seed}_nnheui_pythia_1_4b_sft_full/train_scores.jsonl \\",
        f"  --test-scores  data/scores/mimir_github_nonmember_control_seed{seed}_nnheui_pythia_1_4b_sft_full/test_scores.jsonl \\",
        f"  --output-dir   outputs/runs/nonmember_control_seed{seed}_nnheui_pythia_1_4b_sft_full \\",
        "  --primary-score min_k_20_logprob",
        "",
        "# 6. Generate this report",
        "python scripts/reports/report_nonmember_control.py \\",
        f"  --parent-results outputs/runs/nonmember_control_seed{seed}_pythia_1_4b/results.json \\",
        f"  --target-results outputs/runs/nonmember_control_seed{seed}_nnheui_pythia_1_4b_sft_full/results.json \\",
        f"  --data-manifest data/processed/mimir_github_nonmember_control_seed{seed}/manifest.json \\",
        "  --output-dir outputs/reports/nonmember_vs_nonmember_control_parent_target",
        "```",
        "",
        "---",
        "",
        "## 8. Blockers and Limitations",
        "",
        f"1. **Small pool**: MIMIR GitHub `ngram_13_0.2` nonmember pool has only ~740 examples.",
        f"   With {n_test} test examples per class and {n_train} train per class,",
        f"   the null bound is ≈{null_bound:.3f}.",
        "2. **Single seed**: only seed=0 was run. Multi-seed replication would tighten",
        "   the null estimate (run seeds 1–4 using the same script with `--seed N`).",
        "3. **Threshold-selection noise**: a finite train split means the calibrated",
        "   threshold may not be optimal and can add noise even under the null.",
        "4. **No unrelated-model control**: running an unrelated model (e.g. `gpt2-xl`)",
        "   on the same member/nonmember split would bound the trivial-separability floor.",
        "",
        "---",
        "",
        "## 9. Recommendation",
        "",
    ]

    # Decide overall verdict
    p_ok = abs(p_ctrl_adv) <= null_bound if p_ctrl_adv == p_ctrl_adv else False
    t_ok = abs(t_ctrl_adv) <= null_bound if t_ctrl_adv == t_ctrl_adv else False

    if p_ok and t_ok:
        verdict = (
            "The control advantages are within the finite-sample null bound for both models. "
            "This **supports** the interpretation that the main member-vs-nonmember advantage "
            "(parent 0.380, target 0.375) reflects a genuine membership signal rather than "
            "a generic GitHub-shard artifact."
        )
    elif abs(p_ctrl_adv) <= 0.20 and abs(t_ctrl_adv) <= 0.20:
        verdict = (
            "The control advantages are moderate (≤0.20) and likely within finite-sample noise, "
            "but exceed the theoretical null bound. Run additional seeds (1–4) to confirm. "
            "The main membership signal (parent 0.380, target 0.375) is substantially higher "
            "and likely reflects genuine membership rather than shard artifacts."
        )
    else:
        verdict = (
            "One or both control advantages are HIGH (>0.20). This is suspicious and may indicate "
            "confounding factors such as statistical artifacts in the nonmember pool, "
            "differences in GitHub content statistics between S0 and S1, or a pipeline bug. "
            "Investigate score histograms, token length distributions, and consider running "
            "additional seeds before drawing conclusions from the main experiment."
        )

    lines += [
        verdict,
        "",
        "---",
        "",
        f"*Report generated by `scripts/reports/report_nonmember_control.py` at {now_utc}.*",
    ]

    summary_path = os.path.join(args.output_dir, "summary.md")
    _write_md(lines, summary_path)

    # Print key table to console
    print("\n=== Key Comparison ===")
    print(f"{'Score':<30} {'P_Main':>8} {'P_Ctrl':>8} {'T_Main':>8} {'T_Ctrl':>8}")
    for key in SCORE_KEYS:
        print(
            f"{key:<30} "
            f"{_fmt(MAIN_PARENT_ADV.get(key)):>8} "
            f"{_fmt(parent_res.get(key,{}).get('test',{}).get('shard_advantage')):>8} "
            f"{_fmt(MAIN_TARGET_ADV.get(key)):>8} "
            f"{_fmt(target_res.get(key,{}).get('test',{}).get('shard_advantage')):>8}"
        )
    print(f"\nNull bound (α=0.05, m={n_test}): {null_bound:.4f}")
    print(f"Report: {summary_path}")


if __name__ == "__main__":
    main()
