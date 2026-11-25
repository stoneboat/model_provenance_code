# Raw score diagnostics

Each row of the [distribution table](score_distributions.csv) describes the completion-level scores for one target, shard, split and score. SD is the sample standard deviation with denominator (n-1). AUC and comparisons use the raw score including ties. The [comparison table](paired_score_comparisons.csv) covers all three pairings in the same fixed prompt order. S″ was generated independently and used to train target C.

- **Paired differences.** Subtract the second shard's score from the first for each prompt. The table reports the mean, SD, standard error and an approximate 95% paired t interval across prompts. These are pointwise descriptions of one experiment. The paired win rate compares the two draws for the same prompt and assigns half weight to a tie.
- **Pooled ROC-AUC.** Compare a randomly selected score from the first shard to a randomly selected score from the second over all cross-shard combinations. Higher scores predict the first shard. Ties receive half weight.
- **Distributional separation.** The two-sided empirical Kolmogorov–Smirnov distance is the largest vertical separation of the two empirical CDFs. For a rule that accepts a score strictly above a scalar threshold, the absolute difference in acceptance rates is bounded by this distance, at every threshold. We report the distance without selecting a new detector or changing any decision. No independence-based KS p-value is used for these paired data.
- **Thresholds.** The original fitter recomputed each target's S-versus-S′ threshold on its original 400 construction pairs; it matched the frozen threshold and balanced accuracy exactly. All three shard pairs use that same pre-existing threshold. The original decisions on the 1,600 S/S′ evaluation pairs reproduce exactly. Threshold-based rates on other pairings are descriptive transfers. The secondary MIN-K20 score uses the same procedure.
- **Reproduction.** From the repository root, run the command below after the separate S″ scoring pass. The cached files and frozen experiment remain read-only.

```bash
.cache/vit-venv/bin/python -m experiments.watermark_inheritance.runners.diagnose_q4_distributions \
    --fresh artifacts/m7vv_q4_diagnostics/s_double_prime \
    --out experiments/watermark_inheritance/results/2026-11-25-m7vv-diagnostics \
    --plot
```

If using the local training environment, `PYTHONPATH=/tmp/m7vv-plotdeps` makes its existing matplotlib dependency available for the optional plot.
