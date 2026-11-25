# Frozen MIMIR public controls

qmGv Q1 compares Pythia-1.4B and the independently pretrained GPT-Neo-1.3B
before and after matching three-epoch LoRA fine-tunes on Lamini Docs. qmGv
Q3 and oT8R Q3 additionally compare BioGPT-Large and Pythia-1.4B at step 0
on the same public shard split. All checkpoints use 500 construction and
200 evaluation examples per source, with their pinned native tokenizers.

| Checkpoint | MIN-K20 | Mean log probability |
| --- | ---: | ---: |
| Pythia-1.4B | 0.380 | 0.390 |
| Pythia-1.4B + Lamini | 0.330 | 0.375 |
| GPT-Neo-1.3B | 0.305 | 0.355 |
| GPT-Neo-1.3B + Lamini | 0.275 | 0.330 |
| BioGPT-Large | 0.010 | 0.000 |
| Pythia-1.4B step 0 | 0.060 | 0.045 |

The first four checkpoints cross the 0.1358 independent-sample reference
cutoff for both scores. BioGPT and step 0 stay below it. Public training and
validation splits can differ, so the reference cutoff does not establish a
population false-alarm rate or causal ancestry. [Full-precision table](advantages.csv)

## Reproduce

The [model and split configuration](../../configs/rebuttal_public_controls.json)
pins the four upstream checkpoints and freeze-file hashes. The [Lamini
protocol](../../configs/matched_lamini.yaml) pins `lamini/lamini_docs`,
both parquet files, and the matched training and scoring settings. The
trainer supervises answer and EOS tokens and sees every one of the 1,260
training rows in each of three epochs. Both runs use rank 16, scale 32,
dropout 0.05, seed 42, and 237 optimizer updates. The source model used
for the zero-adapter export check is scored on the same construction split.

Use the exact original `train.jsonl` and `test.jsonl` files. The archived
seed-0 split manifest identifies the source repository, n-gram subset, and
32-word cap. It omits the original dataset revision. The first command below
is a reconstruction attempt from the pinned paper revision and requires
approved dataset access. The GPU runner checks the file hashes before use.

```bash
root=artifacts/rebuttal_public_controls
python scripts/data/prepare_mimir_domain.py \
  --dataset-revision 02500d3b7cece0cb7628e939ba9fc93fdb6362ae \
  --config github --split ngram_13_0.2 --seed 0 --min-words 8 --max-words 32 \
  --num-train-per-class 500 --num-test-per-class 200 --output-dir "$root/prepared"
python experiments/public_pythia_derivatives/runners/run_rebuttal_controls.py \
  --mimir-dir "$root/prepared" --output-dir "$root/run"
```

The original GPU runs used an NVIDIA A100-SXM4-40GB and PyTorch
2.14.1+cu132. Refit the original thresholds and verify the advantages
on CPU from the saved numeric scores.

```bash
python -m experiments.public_pythia_derivatives.reports.verify_rebuttal_controls
```
