# Independent ImageNet-1K DeiT control

qmGv Q3 and oT8R Q2/Q3 compare the ImageNet-1K training and validation
shards on Meta's independently trained `facebook/deit-base-patch16-224`
at revision `fb2c78a54a5637dec350432794f7b93e31f910c9`. The audit uses
the [public ViT preparation and eight-view CLV scorer](../../README.md)
with the matching pinned dataset revision and seed. It fits the
distinguisher on the construction split before evaluation.

The signed held-out advantage is 0.0083, below the level-0.05
reference cutoff 0.04378 for 1,925 images per source. The observed run
used one H200. [Saved summary](observed_summary.json) · [Pinned configuration](../../configs/aistats27_hard_negative.yaml)

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/vit_provenance/configs/aistats27_hard_negative.yaml
mkdir -p experiments/vit_provenance/logs
sbatch experiments/vit_provenance/jobs/run_hard_negative.sbatch
```

A new run verifies the input manifest before scoring. The reported
decision is available in the saved summary. Exact replay from the
original model outputs requires the original per-image score files.
