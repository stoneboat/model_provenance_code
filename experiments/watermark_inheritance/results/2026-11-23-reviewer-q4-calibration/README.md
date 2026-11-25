# Direct-base negative controls and calibration

oT8R Q2 and Q4 extend the §5.1 controls with 12 Dolly and three UltraChat
runs. All start from the pinned unmarked Pythia-1B checkpoint and receive
only downstream fine-tuning. Each run has a distinct training seed and a
distinct paired-audit shard seed. Dolly runs for three epochs. UltraChat
uses one pass over 400,000 distinct examples. Each primary audit uses 2,000
construction pairs and 8,000 held-out prompt pairs.

The primary paired rule flagged **0/15** final checkpoints at α = 0.05.
The largest held-out advantage was 0.01575 and the cutoff was 0.03037.
The one-sided exact 95% upper bound on the false-alarm rate is 18.1%.

The grid varies α over {0.01, 0.025, 0.05, 0.10}, evaluation size over
{250, 500, 1,000, 2,000, 4,000, 8,000}, and control contamination over
{0%, 5%, 10%, 20%, 40%}. Smaller samples use 25 fixed subsamples per
model and configuration, for 375 correlated audits. Contamination copies
the witness-side decision into the control side of selected pairs, whose
differences become zero. The rejection frequency is zero for \(m\geq 500\)
and at most 1/375 for \(m=250\) throughout the reported grid.

[Final checkpoint results](independent_unit_results.csv) ·
[Sensitivity grid](sensitivity_summary.csv) ·
[Per-model grid counts](unit_clustered_sensitivity.csv) ·
[Input hashes and confidence bound](summary.json)

## Reproduce

The [control roster](../../configs/aistats27_false_positive.json) fixes the
model, shard, and watermark seeds. The [calibration configuration](../../configs/aistats27_q4_calibration.json)
fixes the resampling and contamination grid. Follow the upstream model and
dataset pins in the [watermark experiment](../../README.md).

```bash
bash experiments/watermark_inheritance/submit_aistats27_controls.sh
python -m experiments.watermark_inheritance.calibration_analysis \
  --root artifacts/watermark_inheritance/aistats27_reviewer
```

The second command can run on a CPU once the model jobs have written the
endpoint scores and fitted thresholds. The repository includes their recorded
hashes and the compact per-model results. Regenerating the grid from raw pairs
requires the endpoint scores from those training and scoring jobs.
