"""Sweep the held-out shard size m and trace the empirical shard advantage.

Reuses the score files produced by extract_image_scores.py. For each value of
m we subsample m/2 member and m/2 nonmember records from both the MIA train
split (calibration) and the MIA test split (evaluation), refit the threshold
distinguisher, and read the test-split shard advantage. We repeat over
multiple seeds and report mean +/- std.

The critical threshold gamma_alpha = sqrt(log(2/alpha)/m) from the framework's
Lemma is plotted alongside as the level-alpha rejection boundary. The
random-init control is plotted on the same axes as a negative-case reference.

Usage:
    python scripts/experiments/sweep_m_imagenet_shard.py \\
        --positive-run cifar10 \\
        --negative-run random_init \\
        --output-dir outputs/reports/imagenet_shard_m_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.distinguishers import run_distinguisher


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratified_sample(records: list[dict], m_per_class: int, rng: random.Random) -> list[dict]:
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    pos_sample = rng.sample(pos, m_per_class)
    neg_sample = rng.sample(neg, m_per_class)
    sample = pos_sample + neg_sample
    rng.shuffle(sample)
    return sample


def _measure_one(score_key: str, train_recs: list[dict], test_recs: list[dict]) -> dict:
    train_labels = [r["label"] for r in train_recs]
    test_labels = [r["label"] for r in test_recs]
    train_scores = [r[score_key] for r in train_recs]
    test_scores = [r[score_key] for r in test_recs]
    res = run_distinguisher(
        train_labels, train_scores,
        test_labels, test_scores,
        score_name=score_key,
        criterion="balanced_accuracy",
        n_thresholds=2000,
    )
    return {
        "test_advantage": res["test"]["shard_advantage"],
        "test_accuracy": res["test"]["accuracy"],
        "test_auc": res["test"]["auc"],
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep m and plot empirical shard advantage vs critical threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scores-root", default="data/scores")
    p.add_argument("--positive-run", default="cifar10",
                   help="Name of the run with the highest advantage to trace as positive case")
    p.add_argument("--negative-run", default="random_init",
                   help="Name of the negative-control run to trace alongside")
    p.add_argument("--score-key", default="p_on_t_logprob")
    p.add_argument("--m-values", default="20,40,80,160,320,640,1280,1920",
                   help="Comma-separated total shard sizes (m = positives + negatives in held-out split)")
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", default="outputs/reports/imagenet_shard_m_sweep")
    p.add_argument("--prefix", default="imagenet_shard__")
    return p.parse_args()


def main():
    args = parse_args()
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    runs = {
        "positive": args.positive_run,
        "negative": args.negative_run,
    }

    # Load scores
    score_data = {}
    for kind, run in runs.items():
        train = _load_jsonl(os.path.join(args.scores_root, f"{args.prefix}{run}", "train_scores.jsonl"))
        test = _load_jsonl(os.path.join(args.scores_root, f"{args.prefix}{run}", "test_scores.jsonl"))
        n_train_pos = sum(1 for r in train if r["label"] == 1)
        n_test_pos = sum(1 for r in test if r["label"] == 1)
        max_per_class = min(n_train_pos, len(train) - n_train_pos,
                            n_test_pos, len(test) - n_test_pos)
        score_data[kind] = (train, test, max_per_class)
        print(f"[{kind}={run}] train={len(train)} (pos={n_train_pos}) "
              f"test={len(test)} (pos={n_test_pos}) max_m_per_class={max_per_class}")

    # Run sweep
    results: dict = {}
    for kind, run in runs.items():
        train, test, max_per_class = score_data[kind]
        results[kind] = []
        for m_total in m_values:
            m_per_class = m_total // 2
            if m_per_class > max_per_class:
                print(f"  skipping m={m_total} for {kind} (max_per_class={max_per_class})")
                continue
            advs, accs, aucs = [], [], []
            for seed in range(args.n_seeds):
                rng = random.Random(seed * 9176 + 17 * m_per_class)
                tr_sub = _stratified_sample(train, m_per_class, rng)
                te_sub = _stratified_sample(test, m_per_class, rng)
                metrics = _measure_one(args.score_key, tr_sub, te_sub)
                advs.append(metrics["test_advantage"])
                accs.append(metrics["test_accuracy"])
                aucs.append(metrics["test_auc"] or 0.5)
            entry = {
                "m_total": m_total,
                "m_per_class": m_per_class,
                "advantage_mean": float(sum(advs) / len(advs)),
                "advantage_std": float((sum((a - sum(advs)/len(advs))**2 for a in advs)/len(advs))**0.5),
                "accuracy_mean": float(sum(accs) / len(accs)),
                "auc_mean": float(sum(aucs) / len(aucs)),
                "raw_advantages": advs,
                "gamma_0_05": math.sqrt(math.log(2/args.alpha)/m_total),
            }
            results[kind].append(entry)
            print(f"  [{kind}] m={m_total:>5d}  "
                  f"adv={entry['advantage_mean']:+.4f} +/- {entry['advantage_std']:.4f}  "
                  f"gamma={entry['gamma_0_05']:.4f}  "
                  f"reject_rate={sum(1 for a in advs if a > entry['gamma_0_05'])/len(advs):.2f}")

    # Save raw results
    summary = {
        "positive_run": args.positive_run,
        "negative_run": args.negative_run,
        "score_key": args.score_key,
        "alpha": args.alpha,
        "n_seeds": args.n_seeds,
        "m_values": m_values,
        "results": results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {os.path.join(args.output_dir, 'results.json')}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available, skipping figure: {e}")
        return

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax, ax_rej = axes

    # Critical-threshold reference curve, evaluated on a dense grid
    m_grid = np.geomspace(min(m_values) * 0.7, max(m_values) * 1.3, 200)
    gamma_curve = np.sqrt(np.log(2 / args.alpha) / m_grid)
    ax.plot(m_grid, gamma_curve, color="black", linestyle="--", linewidth=2.0,
            label=r"$\gamma_{0.05} = \sqrt{\log(2/\alpha)/m}$ (reject above)")

    # Positive case: mean +/- std
    mvals_pos = np.array([r["m_total"] for r in results["positive"]])
    means_pos = np.array([r["advantage_mean"] for r in results["positive"]])
    stds_pos = np.array([r["advantage_std"] for r in results["positive"]])
    ax.plot(mvals_pos, means_pos, color="C0", marker="o", markersize=8, linewidth=2.4,
            label=f"derivative ({args.positive_run})")
    ax.fill_between(mvals_pos, means_pos - stds_pos, means_pos + stds_pos,
                    color="C0", alpha=0.20)

    # Negative control
    mvals_neg = np.array([r["m_total"] for r in results["negative"]])
    means_neg = np.array([r["advantage_mean"] for r in results["negative"]])
    stds_neg = np.array([r["advantage_std"] for r in results["negative"]])
    ax.plot(mvals_neg, means_neg, color="C3", marker="s", markersize=7, linewidth=2.4,
            label=f"control ({args.negative_run})")
    ax.fill_between(mvals_neg, means_neg - stds_neg, means_neg + stds_neg,
                    color="C3", alpha=0.20)

    # Estimate the crossover m* where the derivative's mean advantage equals gamma_0.05
    if len(mvals_pos) >= 2:
        log_m = np.log(mvals_pos)
        crossings = []
        for i in range(len(mvals_pos) - 1):
            ga = np.sqrt(np.log(2/args.alpha)/mvals_pos[i])
            gb = np.sqrt(np.log(2/args.alpha)/mvals_pos[i + 1])
            ya, yb = means_pos[i], means_pos[i + 1]
            if (ya - ga) * (yb - gb) < 0:
                t = (ga - ya) / ((yb - ya) - (gb - ga))
                m_star = float(np.exp(log_m[i] + t * (log_m[i + 1] - log_m[i])))
                crossings.append(m_star)
        if crossings:
            m_star = crossings[0]
            ax.axvline(m_star, color="C2", linewidth=1.6, alpha=0.7, linestyle=":")
            ax.text(m_star * 1.07, 0.18,
                    f"$m^{{*}} \\approx {int(round(m_star))}$",
                    color="C2", fontsize=15, va="top", ha="left")

    # Zoom in: tighter y-range so the action near gamma_0.05 is visible.
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.10, 0.22)
    ax.set_ylabel(r"Empirical shard advantage  $\widehat{\mathrm{Adv}}_{A}(T,S,S')$")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)

    # Rejection-rate panel
    rej_pos = np.array([
        sum(1 for a in r["raw_advantages"] if a > r["gamma_0_05"]) / len(r["raw_advantages"])
        for r in results["positive"]
    ])
    rej_neg = np.array([
        sum(1 for a in r["raw_advantages"] if a > r["gamma_0_05"]) / len(r["raw_advantages"])
        for r in results["negative"]
    ])
    ax_rej.plot(mvals_pos, rej_pos, color="C0", marker="o", markersize=8, linewidth=2.4,
                label="derivative")
    ax_rej.plot(mvals_neg, rej_neg, color="C3", marker="s", markersize=7, linewidth=2.4,
                label="control")
    ax_rej.axhline(args.alpha, color="black", linestyle="--", linewidth=1.4, alpha=0.8,
                   label=fr"$\alpha = {args.alpha}$")
    ax_rej.set_xscale("log")
    ax_rej.set_xlabel(r"Held-out shard size $m$ (members + nonmembers)")
    ax_rej.set_ylabel("Reject rate")
    ax_rej.set_ylim(-0.05, 1.05)
    ax_rej.legend(loc="center right", framealpha=0.95)
    ax_rej.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "advantage_vs_m.png")
    out_pdf = os.path.join(args.output_dir, "advantage_vs_m.pdf")
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}\nSaved {out_pdf}")


if __name__ == "__main__":
    main()
