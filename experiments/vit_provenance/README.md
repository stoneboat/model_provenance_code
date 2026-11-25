# ViT provenance-witness audits

Section 5.2.2, Table 3, and Appendix D.3 use ImageNet-1K training and
validation shards to audit the parent, six public derivatives, and a
randomly initialized ViT. Appendix D.4 applies the same audit to
transformed encoder checkpoints. The DeiT control supports qmGv Q3 and
oT8R Q2/Q3.

## Public derivatives

The [launcher](jobs/run_public_derivatives.sbatch) pins the parent
`google/vit-base-patch16-224` and six public derivative revisions.
The [preparation](runners/prepare_shards.py) uses
`evanarlian/imagenet_1k_resized_256` at
`8de107e08d1f61884d5a4ec2dea8740667264a60`, with seed 0 and
77 classes. It selects 25 images per class and source for each of the
disjoint construction and evaluation splits. Held-out \(m=1,925\) per
source gives a level-0.05 reference cutoff of 0.0438.
[Audit settings](configs/clv.yaml)

The centered-logit-variance (CLV) scorer queries each target on eight
fixed weak views, centers the log-probability vectors by view, and uses
negative population variance across views averaged over the output
classes. It uses the complete output vectors. Labels are unnecessary
to compute the CLV score.

```bash
mkdir -p experiments/vit_provenance/logs
sbatch experiments/vit_provenance/jobs/run_public_derivatives.sbatch
```

The [submitted Table 8](results/submitted_table_8.csv) reports detections
for all six derivatives, with gaps from 0.1029 to 0.1288. The parent has
0.0810 and the random-initialization control 0.0062. An archived rerun
with the repository's pinned reconstruction gave smaller advantages
and different decisions. Exact replication needs the original image
identities and run artifacts. The launcher writes a new CSV to
`artifacts/vit_provenance/public_derivatives_seed0/reports/`.

## Transformations

[Submitted Table 9](results/submitted_table_9.csv) reports detections for
Linear Soup, TIES, DARE-TIES and SLERP, 8/4/3-bit simulated weight
quantization, and magnitude pruning through 70%. At 90% pruning the gap
was 0.0369. All transformations use the CIFAR-10 derivative as the
baseline and the same ImageNet CLV protocol.

The saved source omits the original transformed checkpoints, the exact
three merge inputs, and complete merge, quantization and pruning settings.
Given an original checkpoint and the prepared shards, run the shared
score and threshold audit on the local checkpoint directory.

```bash
root=artifacts/vit_provenance
data="$root/public_derivatives_seed0/prepared/imagenet_shard"
out="$root/local_transformed_audit"
python experiments/vit_provenance/scoring/score_clv.py \
  --target-model "$TRANSFORMED_CHECKPOINT" --train-file "$data/train.jsonl" \
  --test-file "$data/test.jsonl" --output-dir "$out/scores"
python scripts/audit/run_threshold_audit.py --model-label "Transformed ViT" \
  --train-scores "$out/scores/train_scores.jsonl" --test-scores "$out/scores/test_scores.jsonl" \
  --score-keys clv_score --primary-score clv_score --output-dir "$out/results"
```

## Independent DeiT control

The separately trained ImageNet-1K DeiT-Base checkpoint gives a signed
advantage of 0.0083 against the same 0.0438 cutoff. [Revision, settings, and commands](results/2026-11-23-hard-negative/README.md)
