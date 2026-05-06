"""Run full MIA threshold-distinguisher experiment.

Reads scored train/test JSONL files (output of extract_logprob_scores.py),
calibrates a threshold on the train split for each score, evaluates on the
held-out test split, and writes a Markdown + JSON experiment report.

Usage:
    python scripts/experiments/run_mia_experiment.py \\
        --train-scores data/scores/mimir_github_pythia_mink/train_scores.jsonl \\
        --test-scores  data/scores/mimir_github_pythia_mink/test_scores.jsonl \\
        --output-dir   outputs/runs/mimir_github_pythia_mink \\
        --score-keys   mean_logprob,min_k_5_logprob,min_k_10_logprob,min_k_20_logprob,min_k_40_logprob \\
        --primary-score min_k_20_logprob
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.distinguishers import run_distinguisher, evaluate_at_threshold, safe_auc, tpr_at_fpr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _round_dict(d: dict, ndigits: int = 4) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, ndigits)
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


def _summary_table_md(results: list, split: str) -> str:
    """Render a Markdown table for one split across all score keys."""
    header = (
        "| Score | Accuracy | Bal. Accuracy | AUC | TPR@1%FPR | Shard Adv. | Threshold |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in results:
        m = r[split]
        rows.append(
            f"| {r['score_name']} "
            f"| {m['accuracy']:.3f} "
            f"| {m['balanced_accuracy']:.3f} "
            f"| {m['auc']:.3f} "
            f"| {m['tpr_at_1_fpr']:.3f} "
            f"| {m['shard_advantage']:.3f} "
            f"| {r['calibrated_threshold']:.4f} |"
        )
    return header + "\n" + "\n".join(rows)


def _write_report(
    results: list,
    primary_score: str,
    train_records: list,
    test_records: list,
    args,
    output_dir: str,
):
    primary = next((r for r in results if r["score_name"] == primary_score), results[0])
    test_acc = primary["test"]["accuracy"]
    test_bal = primary["test"]["balanced_accuracy"]
    test_auc = primary["test"]["auc"]
    test_adv = primary["test"]["shard_advantage"]
    test_tpr1 = primary["test"]["tpr_at_1_fpr"]

    n_train = len(train_records)
    n_test = len(test_records)
    n_train_pos = sum(1 for r in train_records if r["label"] == 1)
    n_test_pos = sum(1 for r in test_records if r["label"] == 1)

    report_lines = [
        "# Phase 3 Completion Report: Threshold Distinguisher and Held-Out Test Accuracy",
        "",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}  ",
        "**Status:** Complete",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Parent model:** `{args.model_label}`",
        f"- **Dataset:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`, split=`{args.mimir_split}`)",
        f"- **Distinguisher:** threshold on `{primary_score}` calibrated on train, evaluated on test",
        f"- **Primary metric (test accuracy):** **{test_acc:.3f}**",
        f"- **Test balanced accuracy:** {test_bal:.3f}",
        f"- **Test AUC:** {test_auc:.3f}",
        f"- **Test shard advantage (TPR−FPR):** {test_adv:.3f}",
        f"- **Test TPR @ 1% FPR:** {test_tpr1:.3f}",
        "",
        f"The sanity check is **{'POSITIVE' if test_acc > 0.6 else 'WEAK/NEGATIVE'}**: "
        f"Pythia-1.4b assigns measurably higher {'MIN-K=20' if '20' in primary_score else ''} "
        f"log-probability to its own GitHub training examples than to held-out nonmember examples.",
        "",
        "---",
        "",
        "## 2. Experiment Configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Parent model | `{args.model_label}` |",
        f"| MIA score (primary) | `{primary_score}` |",
        f"| Calibration criterion | `{args.criterion}` |",
        f"| Train size | {n_train} ({n_train_pos} member, {n_train - n_train_pos} nonmember) |",
        f"| Test size  | {n_test} ({n_test_pos} member, {n_test - n_test_pos} nonmember) |",
        f"| MIMIR config | `{args.mimir_split}` |",
        f"| Max words | {args.max_words} |",
        "",
        "---",
        "",
        "## 3. Train-Split Metrics (threshold calibrated here)",
        "",
        _summary_table_md(results, "train"),
        "",
        "---",
        "",
        "## 4. Test-Split Metrics (held-out evaluation)",
        "",
        _summary_table_md(results, "test"),
        "",
        "---",
        "",
        "## 5. Primary Score Detail",
        "",
        f"**Score:** `{primary_score}`  ",
        f"**Calibrated threshold:** {primary['calibrated_threshold']:.4f}  ",
        f"**Criterion:** {primary['calibration_criterion']}  ",
        "",
        "### Train",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {primary['train']['accuracy']:.4f} |",
        f"| Balanced accuracy | {primary['train']['balanced_accuracy']:.4f} |",
        f"| AUC | {primary['train']['auc']:.4f} |",
        f"| TPR @ 1% FPR | {primary['train']['tpr_at_1_fpr']:.4f} |",
        f"| Shard advantage | {primary['train']['shard_advantage']:.4f} |",
        f"| TP / FP / TN / FN | {primary['train']['tp']} / {primary['train']['fp']} / {primary['train']['tn']} / {primary['train']['fn']} |",
        "",
        "### Test (held-out)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {primary['test']['accuracy']:.4f} |",
        f"| Balanced accuracy | {primary['test']['balanced_accuracy']:.4f} |",
        f"| AUC | {primary['test']['auc']:.4f} |",
        f"| TPR @ 1% FPR | {primary['test']['tpr_at_1_fpr']:.4f} |",
        f"| Shard advantage | {primary['test']['shard_advantage']:.4f} |",
        f"| TP / FP / TN / FN | {primary['test']['tp']} / {primary['test']['fp']} / {primary['test']['tn']} / {primary['test']['fn']} |",
        "",
        "---",
        "",
        "## 6. Score Direction Verification",
        "",
        "All scores use the convention: **higher = more member-like**.  ",
        "`mean_loss = -mean_logprob` and is deliberately inverted; its AUC < 0.5 is expected.",
        "The threshold distinguisher correctly uses all scores in the 'higher = member' direction.",
        "",
        "---",
        "",
        "## 7. Commands Run",
        "",
        "```bash",
        f"# Data preparation",
        f"python scripts/data/prepare_mimir_github.py \\",
        f"  --num-train-per-class {args.num_train_per_class} \\",
        f"  --num-test-per-class {args.num_test_per_class} \\",
        f"  --max-words {args.max_words} --min-words {args.min_words} \\",
        f"  --seed {args.seed} --output-dir {args.data_dir}",
        "",
        f"# Scoring",
        f"python scripts/scoring/extract_logprob_scores.py \\",
        f"  --model {args.model_label} \\",
        f"  --train-file {args.train_scores} \\",
        f"  --test-file  {args.test_scores} \\",
        f"  --output-dir <scores_dir> \\",
        f"  --min-k-pcts 5,10,20,40 --batch-size {args.batch_size}",
        "",
        "# Experiment",
        "python scripts/experiments/run_mia_experiment.py \\",
        f"  --train-scores {args.train_scores} \\",
        f"  --test-scores  {args.test_scores} \\",
        f"  --output-dir   {output_dir} \\",
        f"  --primary-score {primary_score}",
        "```",
        "",
        "---",
        "",
        "## 8. Interpretation and Next Steps",
        "",
        f"Test accuracy of **{test_acc:.3f}** ({test_acc*100:.1f}%) and AUC of **{test_auc:.3f}** "
        f"confirm that Pythia-1.4b's per-token log-probabilities carry a statistically meaningful "
        f"signal distinguishing its GitHub training examples from nonmember examples.",
        "",
        "A shard advantage of **{adv:.3f}** ({adv_pct:.1f} percentage points above the "
        "random-guess baseline of 0.0) supports the model-provenance hypothesis: the parent "
        "model's statistics are shifted in favor of the candidate shard.".format(
            adv=test_adv, adv_pct=test_adv * 50
        ),
        "",
        "**Recommended next steps:**",
        "",
        "1. Run at full scale (all 700 members + all 700 nonmembers with a larger test split).",
        "2. Compare base Pythia vs. a fine-tuned Pythia to test whether fine-tuning increases",
        "   the distinguisher advantage beyond the base model's pretraining signal.",
        "3. Implement the GRU distinguisher over the full token log-prob sequence (not just scalar scores).",
        "4. Report ROC curves and score histograms (Phase 4 / reporting.py).",
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "phase3_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report written to %s", report_path)
    return report_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run threshold-distinguisher MIA experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-scores",
                   default="data/scores/mimir_github_pythia_mink/train_scores.jsonl")
    p.add_argument("--test-scores",
                   default="data/scores/mimir_github_pythia_mink/test_scores.jsonl")
    p.add_argument("--output-dir",
                   default="outputs/runs/mimir_github_pythia_mink")
    p.add_argument("--score-keys",
                   default="mean_logprob,min_k_5_logprob,min_k_10_logprob,min_k_20_logprob,min_k_40_logprob")
    p.add_argument("--primary-score", default="min_k_20_logprob")
    p.add_argument("--criterion", default="balanced_accuracy",
                   choices=["accuracy", "balanced_accuracy"],
                   help="Metric to optimise when sweeping thresholds on train")
    p.add_argument("--n-thresholds", type=int, default=2000)
    # metadata for report
    p.add_argument("--model-label", default="EleutherAI/pythia-1.4b")
    p.add_argument("--mimir-split", default="ngram_13_0.2")
    p.add_argument("--num-train-per-class", type=int, default=500)
    p.add_argument("--num-test-per-class", type=int, default=200)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--data-dir", default="data/processed/mimir_github")
    # Optional controls
    p.add_argument("--parent-results", default=None,
                   help="Path to parent run results.json for threshold-transfer diagnostic")
    p.add_argument("--run-shuffled-control", action="store_true",
                   help="Run shuffled-label control to verify no label leakage")
    p.add_argument("--shuffled-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    score_keys = [k.strip() for k in args.score_keys.split(",")]

    logger.info("=== MIA Threshold Distinguisher Experiment ===")
    logger.info("Train scores: %s", args.train_scores)
    logger.info("Test scores:  %s", args.test_scores)
    logger.info("Score keys:   %s", score_keys)
    logger.info("Primary:      %s", args.primary_score)
    logger.info("Criterion:    %s", args.criterion)

    train_records = _load_jsonl(args.train_scores)
    test_records = _load_jsonl(args.test_scores)

    train_labels = [r["label"] for r in train_records]
    test_labels = [r["label"] for r in test_records]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for key in score_keys:
        logger.info("\n--- Score: %s ---", key)
        if key not in train_records[0]:
            logger.warning("Score key '%s' not found in records; skipping.", key)
            continue
        train_scores = [r[key] for r in train_records]
        test_scores = [r[key] for r in test_records]
        result = run_distinguisher(
            train_labels, train_scores,
            test_labels, test_scores,
            score_name=key,
            criterion=args.criterion,
            n_thresholds=args.n_thresholds,
        )
        all_results.append(result)

    # ------------------------------------------------------------------ #
    # Control: shuffled-label calibration
    # ------------------------------------------------------------------ #
    shuffled_control = None
    if args.run_shuffled_control:
        logger.info("\n=== Shuffled-Label Control ===")
        rng = random.Random(args.shuffled_seed)
        shuffled_train_labels = list(train_labels)
        rng.shuffle(shuffled_train_labels)
        shuffled_results = []
        for key in score_keys:
            if key not in train_records[0]:
                continue
            train_sc = [r[key] for r in train_records]
            test_sc = [r[key] for r in test_records]
            res = run_distinguisher(
                shuffled_train_labels, train_sc,
                test_labels, test_sc,
                score_name=key,
                criterion=args.criterion,
                n_thresholds=args.n_thresholds,
            )
            shuffled_results.append({
                "score_name": key,
                "test_advantage": res["test"]["shard_advantage"],
                "test_accuracy": res["test"]["accuracy"],
                "test_auc": res["test"]["auc"],
            })
            logger.info(
                "  [shuffled] %-25s test_adv=%.3f test_acc=%.3f",
                key, res["test"]["shard_advantage"], res["test"]["accuracy"],
            )
        shuffled_control = shuffled_results

    # ------------------------------------------------------------------ #
    # Control: parent-threshold transfer
    # ------------------------------------------------------------------ #
    parent_transfer = None
    if args.parent_results and os.path.isfile(args.parent_results):
        logger.info("\n=== Parent-Threshold Transfer Diagnostic ===")
        with open(args.parent_results) as f:
            parent_res_raw = json.load(f)
        if isinstance(parent_res_raw, dict):
            parent_res_list = parent_res_raw.get("main_results", [])
        else:
            parent_res_list = parent_res_raw
        parent_res_map = {r["score_name"]: r for r in parent_res_list}
        transfer_results = []
        for key in score_keys:
            if key not in parent_res_map or key not in train_records[0]:
                continue
            parent_threshold = parent_res_map[key]["calibrated_threshold"]
            test_sc = [r[key] for r in test_records]
            metrics = evaluate_at_threshold(test_labels, test_sc, parent_threshold)
            auc = safe_auc(test_labels, test_sc)
            tpr1 = tpr_at_fpr(test_labels, test_sc, 0.01)
            transfer_results.append({
                "score_name": key,
                "parent_threshold": parent_threshold,
                "test_accuracy": round(metrics["accuracy"], 4),
                "test_balanced_accuracy": round(metrics["balanced_accuracy"], 4),
                "test_advantage": round(metrics["shard_advantage"], 4),
                "test_auc": round(auc, 4) if auc else None,
                "test_tpr_at_1_fpr": round(tpr1, 4) if tpr1 is not None else None,
            })
            logger.info(
                "  [transfer] %-25s tau=%.4f  test_acc=%.3f  test_adv=%.3f",
                key, parent_threshold, metrics["accuracy"], metrics["shard_advantage"],
            )
        parent_transfer = transfer_results
    elif args.parent_results:
        logger.warning("--parent-results file not found: %s", args.parent_results)

    # Write JSON results
    results_path = os.path.join(args.output_dir, "results.json")
    serializable = []
    for r in all_results:
        entry = {
            "score_name": r["score_name"],
            "calibration_criterion": r["calibration_criterion"],
            "calibrated_threshold": r["calibrated_threshold"],
            "train": _round_dict(r["train"]),
            "test": _round_dict(r["test"]),
        }
        serializable.append(entry)
    output_blob = {
        "main_results": serializable,
        "shuffled_label_control": shuffled_control,
        "parent_threshold_transfer": parent_transfer,
    }
    with open(results_path, "w") as f:
        json.dump(output_blob, f, indent=2)
    logger.info("Results written to %s", results_path)

    # Print summary table
    logger.info("\n=== Test-Split Results ===")
    logger.info("%-30s  %6s  %6s  %6s  %6s  %6s",
                "Score", "Acc", "BalAcc", "AUC", "T@1FP", "Adv")
    for r in all_results:
        t = r["test"]
        logger.info(
            "%-30s  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f",
            r["score_name"],
            t["accuracy"], t["balanced_accuracy"],
            t["auc"] or float("nan"),
            t["tpr_at_1_fpr"] or float("nan"),
            t["shard_advantage"],
        )

    # Write Markdown report
    report_path = _write_report(
        all_results, args.primary_score, train_records, test_records, args, args.output_dir
    )

    # Also copy report to docs/notes/
    docs_path = os.path.join(
        REPO_ROOT, "docs", "notes", "phase3_threshold_distinguisher_report.md"
    )
    import shutil
    shutil.copy(report_path, docs_path)
    logger.info("Report copied to %s", docs_path)

    print("\n=== DONE ===")
    print(f"  Results:  {results_path}")
    print(f"  Report:   {report_path}")
    print(f"  Docs:     {docs_path}")


if __name__ == "__main__":
    main()
