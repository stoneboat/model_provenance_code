"""Post hoc distribution summaries with the original S/S′ detector held fixed."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t

from experiments.watermark_inheritance.score import calibrate_threshold

TARGETS = ("A", "B", "C")
SCORES = {"watermark": "z", "min_k20": "min_k20"}
SIDES = ("s", "s_prime", "s_double_prime")
SPLITS = {"calibration": 400, "audit": 1600}


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def compare(x, y, threshold):
    """Summarize two paired score samples, reporting no data-chosen cutoff."""
    n = len(x)
    paired = x - y
    mean = np.mean(paired)
    sd = np.std(paired, ddof=1)
    se = sd / np.sqrt(n)
    width = t.ppf(.975, df=n-1) * se
    # Mann-Whitney AUC compares every ordered cross-shard combination, including half credit for ties.
    ranks = rankdata(np.r_[x, y], method="average")[:n]
    auc = (ranks.sum() - n*(n+1)/2) / n**2
    # Supremum distance between the two empirical CDFs, a bound on any strict-threshold gap.
    grid = np.unique(np.r_[x, y])
    cdf_x = np.searchsorted(np.sort(x), grid, side="right") / n
    cdf_y = np.searchsorted(np.sort(y), grid, side="right") / n
    gap = ((x > threshold).sum() - (y > threshold).sum()) / n
    return dict(
        paired_mean_difference=mean, paired_sd=sd, paired_se=se,
        paired_mean_95_low=mean-width, paired_mean_95_high=mean+width,
        paired_win_rate_with_half_ties=(np.mean(x > y) + 0.5*np.mean(x == y)),
        pooled_roc_auc=auc, empirical_ks_distance=np.max(np.abs(cdf_x-cdf_y)),
        frozen_threshold_gap=gap, frozen_threshold_balanced_accuracy=(1+gap)/2,
    )


def summarize(target, cache, fresh, marginals, comparisons, plot_data):
    directory = cache / "scores" / target
    rows = read_rows(directory / "examples.jsonl")
    pairs = {row["pair_id"]: row for row in rows}
    frozen = json.loads((directory / "thresholds.json").read_text())["detectors"]
    metrics = json.loads((directory / "metrics.json").read_text())["shard"]
    if len(pairs) != len(rows) or len(rows) != sum(SPLITS.values()):
        raise ValueError("Unexpected cached audit rows")
    sides = SIDES[:2] if fresh is None else SIDES
    if fresh is not None:
        third = read_rows(fresh / f"{target}.jsonl")
        if len(third) != len(rows) or {r["pair_id"] for r in third} != pairs.keys():
            raise ValueError("S″ and S/S′ must cover the same unique pairs")
        for new in third:
            row = pairs[new["pair_id"]]
            if row["audit_split"] != new["audit_split"]:
                raise ValueError("Pair split changed")
            row["s_double_prime"] = new
    for split, n in SPLITS.items():
        subset = [row for row in rows if row["audit_split"] == split]
        if len(subset) != n:
            raise ValueError("Unexpected split size")
        for metric, field in SCORES.items():
            arrays = {s: np.array([row[s][field] for row in subset], dtype=float) for s in sides}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Missing or nonfinite score")
            if split == "calibration":
                recomputed = calibrate_threshold(arrays["s"].tolist(), arrays["s_prime"].tolist())
                if recomputed != frozen[metric]:
                    raise ValueError("Could not reproduce original construction threshold")
            threshold = frozen[metric]["threshold"]
            construction_ba = frozen[metric]["balanced_accuracy"]
            header = dict(target=target, split=split, metric=metric, n_pairs=n)
            for s, a in arrays.items():
                marginals.append(dict(**header, shard=s, mean=a.mean(), sample_sd=a.std(ddof=1),
                    q05=np.quantile(a, .05), median=np.median(a), q95=np.quantile(a, .95),
                    frozen_s_vs_s_prime_threshold=threshold, accepted=int((a>threshold).sum())))
            for x, y in combinations(sides, 2):
                result = compare(arrays[x], arrays[y], threshold)
                if (split == "audit" and (x,y) == ("s","s_prime")
                        and not np.isclose(result["frozen_threshold_gap"],metrics[metric]["signed_gap"])):
                    raise ValueError("Original held-out gap disagrees with cached decisions")
                comparisons.append(dict(**header, positive_shard=x, negative_shard=y,
                    frozen_s_vs_s_prime_threshold=threshold, fitted_s_vs_s_prime_construction_ba=construction_ba,
                    **result))
            if split == "audit":
                plot_data[target, metric] = (arrays, threshold)


def plot_ecdfs(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2,3,figsize=(10,5.7),sharex="row",sharey=True,layout="constrained")
    colors = dict(zip(SIDES, ["#0072B2","#D55E00","#009E73"]))
    labels = dict(zip(SIDES, ["S","S′","S″"]))
    for j, target in enumerate(TARGETS):
        for i, metric in enumerate(SCORES):
            ax = axes[i,j]
            arrays, threshold = data[target,metric]
            for s, a in arrays.items():
                ax.step(np.sort(a),np.arange(1,len(a)+1)/len(a),where="post",
                        color=colors[s],label=labels[s],linewidth=1.3)
            ax.axvline(threshold, color="#505050",linestyle=":",linewidth=1.1)
            if i==0:
                ax.set_title("Target "+target)
            if j==0:
                ax.set_ylabel("Cumulative fraction")
            ax.set_xlabel("Watermark z-score" if i==0 else "MIN-K20 log probability")
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color="#505050",linestyle=":"))
    fig.legend(handles, labels+["Frozen S vs. S′ threshold"],loc="outside lower center",
               ncol=len(handles),frameon=False)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,default=Path("artifacts/m7vv_q4"))
    ap.add_argument("--fresh",type=Path,help="Optional directory of A/B/C.jsonl scored on S″")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--plot",action="store_true")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    marginals, comparisons, plot_data = [], [], {}
    for target in TARGETS:
        summarize(target,args.cache,args.fresh,marginals,comparisons,plot_data)
    write_csv(args.out/"score_distributions.csv", marginals)
    write_csv(args.out/"paired_score_comparisons.csv", comparisons)
    if args.plot:
        plot_ecdfs(plot_data,args.out/"heldout_ecdfs.png")


if __name__ == "__main__":
    main()
