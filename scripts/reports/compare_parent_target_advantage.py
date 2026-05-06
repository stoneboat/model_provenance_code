"""Compare parent and target model shard-membership advantage.

Reads results.json from a parent run and a target run, computes
advantage preservation ratios, and writes a Markdown comparison report.

Usage:
    python scripts/reports/compare_parent_target_advantage.py \\
        --parent-results outputs/runs/mimir_github_pythia_mink/results.json \\
        --target-results outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full/results.json \\
        --parent-model EleutherAI/pythia-1.4b \\
        --target-model nnheui/pythia-1.4b-sft-full \\
        --target-ft-dataset HuggingFaceH4/ultrachat_200k \\
        --output-dir outputs/reports/target_membership_advantage_nnheui_pythia_1_4b_sft_full
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Support both old (list) and new (dict) format
    if isinstance(data, list):
        return {"main_results": data}
    return data


def _adv_ratio(target_adv: float, parent_adv: float) -> str:
    if parent_adv == 0:
        return "inf" if target_adv != 0 else "—"
    ratio = target_adv / parent_adv
    return f"{ratio:.3f}"


def _interpret_ratio(target_adv: float, parent_adv: float) -> str:
    if parent_adv == 0:
        return "parent advantage is zero; ratio undefined"
    ratio = target_adv / parent_adv
    if ratio >= 0.75:
        return "strong preservation (≥75%)"
    if ratio >= 0.25:
        return "partial preservation (25–75%)"
    if ratio > 0.05:
        return "weak / attenuated (<25%)"
    return "near-zero — no detectable inherited signal"


def _build_summary(
    parent_results: dict,
    target_results: dict,
    args,
    output_dir: str,
) -> str:
    parent_main = {r["score_name"]: r for r in parent_results["main_results"]}
    target_main = {r["score_name"]: r for r in target_results["main_results"]}
    shuffled = target_results.get("shuffled_label_control") or []
    transfer = target_results.get("parent_threshold_transfer") or []

    score_keys = list(parent_main.keys())

    # Primary score for headline
    primary = args.primary_score
    p_prim = parent_main.get(primary, {})
    t_prim = target_main.get(primary, {})
    p_adv = p_prim.get("test", {}).get("shard_advantage", float("nan"))
    t_adv = t_prim.get("test", {}).get("shard_advantage", float("nan"))
    t_acc = t_prim.get("test", {}).get("accuracy", float("nan"))
    t_auc = t_prim.get("test", {}).get("auc", float("nan"))
    t_tpr1 = t_prim.get("test", {}).get("tpr_at_1_fpr", float("nan"))
    t_bal = t_prim.get("test", {}).get("balanced_accuracy", float("nan"))

    # --- Target scoring diagnostics from manifest ---
    target_scores_dir = os.path.dirname(args.target_results)
    scoring_manifest = {}
    manifest_path = os.path.join(target_scores_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            scoring_manifest = json.load(f)

    n_train_scored = scoring_manifest.get("n_train_scored", "?")
    n_test_scored = scoring_manifest.get("n_test_scored", "?")
    device = scoring_manifest.get("device", "?")
    dtype = scoring_manifest.get("dtype", "?")
    batch_size = scoring_manifest.get("batch_size", "?")
    tok_hist = scoring_manifest.get("token_length_histogram", {})
    tok_hist_str = ", ".join(f"{k}: {v}" for k, v in sorted(tok_hist.items())) if tok_hist else "?"

    # --- Build report lines ---
    lines = [
        "# Target-Model Shard-Membership Advantage Report",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Target model:** `{args.target_model}`",
        f"- **Claimed parent:** `{args.parent_model}`",
        f"- **Fine-tuning dataset:** `{args.target_ft_dataset}`",
        f"- **Dataset/shard:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`, split=`{args.mimir_split}`)",
        f"- **Primary score:** `{primary}`",
        f"- **Target test accuracy:** {t_acc:.3f}",
        f"- **Target balanced accuracy:** {t_bal:.3f}",
        f"- **Target AUC:** {t_auc:.3f}",
        f"- **Target shard advantage (TPR−FPR):** {t_adv:.3f}",
        f"- **Target TPR @ 1% FPR:** {t_tpr1:.3f}",
        f"- **Parent shard advantage:** {p_adv:.3f}",
        f"- **Advantage preservation ratio:** {_adv_ratio(t_adv, p_adv)} ({_interpret_ratio(t_adv, p_adv)})",
        "",
        "**Main conclusion:**",
        "",
        f"The target model `{args.target_model}` shows a shard advantage of "
        f"**{t_adv:.3f}** on the MIMIR GitHub member/nonmember split, compared to the "
        f"parent's **{p_adv:.3f}**. Preservation ratio: **{_adv_ratio(t_adv, p_adv)}**.",
        f"Interpretation: {_interpret_ratio(t_adv, p_adv)}.",
        "",
        "---",
        "",
        "## 2. Model Metadata",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Parent model | `{args.parent_model}` |",
        f"| Target model | `{args.target_model}` |",
        f"| Target model card base | `EleutherAI/pythia-1.4b` (from `base_model:finetune` tag) |",
        f"| Fine-tuning dataset | `{args.target_ft_dataset}` |",
        "| Architecture class | `GPTNeoXForCausalLM` |",
        "| Model type | `gpt_neox` |",
        "| Tokenizer | `GPTNeoXTokenizerFast` |",
        "| Vocab size | 50,304 (same as Pythia-1.4b) |",
        "| Pad token | `<\\|endoftext\\|>` (pre-set in tokenizer config) |",
        "| Chat template | present but **not used** (raw text scored directly) |",
        "",
        "---",
        "",
        "## 3. Data and Label Semantics",
        "",
        f"- **MIMIR config:** `github`",
        f"- **MIMIR n-gram split:** `{args.mimir_split}`",
        "- **Member label (1) meaning:** text drawn from Pythia's GitHub pretraining corpus",
        "- **Nonmember label (0) meaning:** text NOT in Pythia's GitHub pretraining corpus",
        "- **Labels are fixed** to pretraining membership in `EleutherAI/pythia-1.4b`; they do NOT",
        "  reflect whether the text was in the UltraChat fine-tuning dataset.",
        f"- **Train size:** {n_train_scored} (500 member + 500 nonmember)",
        f"- **Test size:** {n_test_scored} (200 member + 200 nonmember)",
        "- **Record hash match with parent run:** YES — same `data/processed/mimir_github/` files used.",
        "- **Preprocessing:** 32-word truncation, min 8 words, seed=0",
        "",
        "---",
        "",
        "## 4. Target Scoring Diagnostics",
        "",
        f"- **Device:** `{device}`",
        f"- **Dtype:** `{dtype}`",
        f"- **Batch size:** {batch_size}",
        f"- **Train examples scored:** {n_train_scored}",
        f"- **Test examples scored:** {n_test_scored}",
        f"- **Token length distribution:** {tok_hist_str}",
        "- **Causal-shift:** identical logic to parent — `shift_logits = logits[:, :-1]`,",
        "  first token not scored, padding excluded via attention mask.",
        "- **No chat template applied:** texts scored as raw 32-word GitHub snippets.",
        "",
        "---",
        "",
        "## 5. Target-Calibrated Distinguisher Results",
        "",
        "| Score | Test Acc | Bal Acc | AUC | TPR@1%FPR | Shard Adv | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key in score_keys:
        t = target_main.get(key, {})
        tm = t.get("test", {})
        lines.append(
            f"| `{key}` "
            f"| {tm.get('accuracy', float('nan')):.3f} "
            f"| {tm.get('balanced_accuracy', float('nan')):.3f} "
            f"| {(tm.get('auc') or float('nan')):.3f} "
            f"| {(tm.get('tpr_at_1_fpr') or float('nan')):.3f} "
            f"| {tm.get('shard_advantage', float('nan')):.3f} "
            f"| {t.get('calibrated_threshold', float('nan')):.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Parent vs Target Comparison",
        "",
        "| Score | Parent Adv | Target Adv | Adv Ratio | Parent AUC | Target AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for key in score_keys:
        p = parent_main.get(key, {})
        t = target_main.get(key, {})
        p_a = p.get("test", {}).get("shard_advantage", float("nan"))
        t_a = t.get("test", {}).get("shard_advantage", float("nan"))
        p_u = p.get("test", {}).get("auc", float("nan"))
        t_u = t.get("test", {}).get("auc", float("nan")) or float("nan")
        ratio = _adv_ratio(t_a, p_a)
        lines.append(
            f"| `{key}` | {p_a:.3f} | {t_a:.3f} | {ratio} | {p_u:.3f} | {t_u:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 7. Parent-Threshold Transfer Diagnostic",
        "",
    ]

    if transfer:
        lines += [
            "Applying the parent-calibrated threshold directly to target test scores:",
            "",
            "| Score | Parent Threshold | Target Test Acc | Target Adv | Target AUC |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in transfer:
            lines.append(
                f"| `{r['score_name']}` "
                f"| {r['parent_threshold']:.4f} "
                f"| {r['test_accuracy']:.3f} "
                f"| {r['test_advantage']:.3f} "
                f"| {(r.get('test_auc') or float('nan')):.3f} |"
            )
        lines += [
            "",
            "Fine-tuning can shift the absolute log-probability scale, so the transferred",
            "threshold may not be optimal. Non-zero advantage here nonetheless indicates",
            "that the parent score distribution and target score distribution are correlated.",
        ]
    else:
        lines += [
            "Parent-threshold transfer was not run for this report.",
            "Run with `--parent-results <path>` on the target experiment to enable.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 8. Controls",
        "",
    ]

    # Shuffled label control
    if shuffled:
        lines += ["### Shuffled-Label Calibration", ""]
        lines += [
            "Labels were randomly permuted in the train split before threshold selection,",
            "then the calibrated threshold was applied to the true test labels.",
            "Expected: test advantage ≈ 0.",
            "",
            "| Score | Shuffled Test Adv | Shuffled Test Acc |",
            "|---|---:|---:|",
        ]
        for r in shuffled:
            lines.append(
                f"| `{r['score_name']}` "
                f"| {r['test_advantage']:.3f} "
                f"| {r['test_accuracy']:.3f} |"
            )
        max_shuf = max(abs(r["test_advantage"]) for r in shuffled)
        lines += [
            "",
            f"Maximum shuffled advantage: **{max_shuf:.3f}** "
            f"({'near zero ✓' if max_shuf < 0.05 else 'non-zero — investigate'})",
        ]
    else:
        lines += [
            "### Shuffled-Label Calibration",
            "",
            "Not run for this report. Run with `--run-shuffled-control` to enable.",
        ]

    lines += [
        "",
        "### Parent Replay",
        "",
        "The parent model (`EleutherAI/pythia-1.4b`) was scored in Phase 3 on the same",
        "data files. Results are used directly without re-scoring.",
        "",
        f"| Score | Replayed Adv | Replayed AUC |",
        "|---|---:|---:|",
    ]
    for key in ["mean_logprob", "min_k_20_logprob", "min_k_40_logprob"]:
        p = parent_main.get(key, {})
        pm = p.get("test", {})
        lines.append(
            f"| `{key}` | {pm.get('shard_advantage', float('nan')):.3f} "
            f"| {(pm.get('auc') or float('nan')):.3f} |"
        )

    lines += [
        "",
        "### Unrelated Model Control",
        "",
        "Not run in this phase. Recommended: score `gpt2-xl` or `EleutherAI/gpt-neo-1.3B`",
        "on the same shard to establish a baseline advantage for models not derived from Pythia.",
        "",
        "### UltraChat / MIMIR Overlap Check",
        "",
        "Not performed in this phase. Exact-hash and n-gram overlap between MIMIR GitHub",
        "texts and `HuggingFaceH4/ultrachat_200k` was not audited.",
        "**Limitation:** if MIMIR GitHub code snippets appear verbatim in UltraChat,",
        "the target model may have *directly* seen shard texts during fine-tuning,",
        "which would weaken the inherited-provenance interpretation.",
        "",
        "---",
        "",
        "## 9. Interpretation",
        "",
        "### Statistical claim",
        "",
        f"The target model `{args.target_model}` assigns statistically distinguishable",
        "log-probabilities to MIMIR GitHub *member* examples versus *nonmember* examples.",
        f"The best test AUC across scores is "
        f"**{max((target_main.get(k, {}).get('test', {}).get('auc') or 0) for k in score_keys):.3f}**,",
        "which is well above the 0.5 random-chance baseline.",
        "",
        "### Provenance claim",
        "",
        "The preservation ratio indicates that the target **retains a substantial fraction**",
        "of the parent's pretraining membership signal.",
        "",
        "**Assumptions required for a causal provenance claim:**",
        "",
        "1. The target was fine-tuned from `EleutherAI/pythia-1.4b` (confirmed by model card tags).",
        "2. The target could not have *independently* acquired the MIMIR GitHub membership",
        "   signal without inheriting it from the parent (requires UltraChat overlap check).",
        "",
        "**Caveats:**",
        "",
        "- UltraChat fine-tuning data overlap with MIMIR GitHub was not audited.",
        "- Fine-tuning can attenuate or amplify pretraining memorization in unpredictable ways.",
        "- A scalar threshold distinguisher is less sensitive than a sequence-level GRU.",
        "- The provenance claim requires additional causal assumptions beyond the statistical result.",
        "",
        "---",
        "",
        "## 10. Commands Run",
        "",
        "```bash",
        "# Target scoring",
        f"python scripts/scoring/extract_logprob_scores.py \\",
        f"  --model {args.target_model} \\",
        f"  --train-file data/processed/mimir_github/train.jsonl \\",
        f"  --test-file  data/processed/mimir_github/test.jsonl \\",
        f"  --output-dir data/scores/mimir_github_nnheui_pythia_1_4b_sft_full \\",
        f"  --min-k-pcts 5,10,20,40 --batch-size {batch_size}",
        "",
        "# Target experiment",
        "python scripts/experiments/run_mia_experiment.py \\",
        f"  --train-scores data/scores/mimir_github_nnheui_pythia_1_4b_sft_full/train_scores.jsonl \\",
        f"  --test-scores  data/scores/mimir_github_nnheui_pythia_1_4b_sft_full/test_scores.jsonl \\",
        f"  --output-dir   outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full \\",
        f"  --primary-score {primary} \\",
        f"  --parent-results outputs/runs/mimir_github_pythia_mink/results.json \\",
        f"  --run-shuffled-control",
        "",
        "# Comparison report",
        "python scripts/reports/compare_parent_target_advantage.py \\",
        f"  --parent-results outputs/runs/mimir_github_pythia_mink/results.json \\",
        f"  --target-results outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full/results.json \\",
        f"  --output-dir {output_dir}",
        "```",
        "",
        "---",
        "",
        "## 11. Blockers and Limitations",
        "",
        "1. **UltraChat overlap not audited** — see Section 8.",
        "2. **No unrelated-model control** — running `gpt2-xl` would establish whether the",
        "   member/nonmember split is trivially easy for any large LM.",
        "3. **Scalar threshold only** — a GRU over the full token log-prob sequence would be",
        "   more sensitive and would better distinguish inherited vs. incidental signal.",
        "",
        "---",
        "",
        "## 12. Recommended Next Step",
        "",
        "In order of priority:",
        "",
        f"1. **Run UltraChat overlap check** to validate the provenance interpretation.",
        f"2. **Run unrelated-model control** (`gpt2-xl` or `EleutherAI/gpt-neo-1.3B`) to",
        "   establish whether the shard is inherently distinguishable independent of Pythia.",
        f"3. **Scale to Pythia-6.9B** (parent) + a 6.9B-derived fine-tuned target to test",
        "   whether the advantage scales with model size.",
        f"4. **Implement GRU distinguisher** (`src/shard_audit/distinguishers.py`) over the full",
        "   per-token log-prob sequence for a stronger provenance signal.",
        f"5. **Try deduped-parent lineage**: re-run parent sanity check with",
        "   `EleutherAI/pythia-1.4b-deduped`, then score a deduped-derived target.",
    ]

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare parent and target shard-membership advantage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parent-results",
                   default="outputs/runs/mimir_github_pythia_mink/results.json")
    p.add_argument("--target-results",
                   default="outputs/runs/mimir_github_nnheui_pythia_1_4b_sft_full/results.json")
    p.add_argument("--parent-model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--target-model", default="nnheui/pythia-1.4b-sft-full")
    p.add_argument("--target-ft-dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--mimir-split", default="ngram_13_0.2")
    p.add_argument("--primary-score", default="min_k_20_logprob")
    p.add_argument("--output-dir",
                   default="outputs/reports/target_membership_advantage_nnheui_pythia_1_4b_sft_full")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading parent results: %s", args.parent_results)
    parent_results = _load_results(args.parent_results)
    logger.info("Loading target results: %s", args.target_results)
    target_results = _load_results(args.target_results)

    os.makedirs(args.output_dir, exist_ok=True)

    report_text = _build_summary(parent_results, target_results, args, args.output_dir)

    out_path = os.path.join(args.output_dir, "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Summary written to %s", out_path)

    # Print key metrics
    target_main = {r["score_name"]: r for r in target_results["main_results"]}
    parent_main = {r["score_name"]: r for r in parent_results["main_results"]}
    logger.info("\n=== Key Comparison ===")
    logger.info("%-30s  %8s  %8s  %8s", "Score", "P_Adv", "T_Adv", "Ratio")
    for key in target_main:
        p_a = parent_main.get(key, {}).get("test", {}).get("shard_advantage", float("nan"))
        t_a = target_main.get(key, {}).get("test", {}).get("shard_advantage", float("nan"))
        ratio = f"{t_a/p_a:.3f}" if p_a != 0 else "inf"
        logger.info("%-30s  %8.3f  %8.3f  %8s", key, p_a, t_a, ratio)

    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
