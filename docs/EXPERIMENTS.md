# Experiment inventory

This inventory follows the submitted AISTATS 2027 manuscript and the final
author responses for Za6d, qmGv, oT8R, m7VV, and the Area Chair. The meta-review
response summarizes the reviewer-facing experiments and adds no new
measurements.

| Reported result | Implementation and configuration | Preparation | Saved results |
| --- | --- | --- | --- |
| §5.1 and Table 1, Dolly and UltraChat final targets and controls. Six detections among descendants and zero among six controls. | [Dolly launch](../experiments/watermark_inheritance/submit.sh), [UltraChat launch](../experiments/watermark_inheritance/submit_ultrachat.sh), [pinned Pythia-1B recipe](../experiments/watermark_inheritance/configure.py), [watermark scorer](../experiments/watermark_inheritance/score.py). | [WikiText and Dolly](../experiments/watermark_inheritance/prepare.py), [watermark pairing](../experiments/watermark_inheritance/generate.py), [UltraChat](../experiments/watermark_inheritance/curve_prepare.py). | [Table 1 and protocol](../experiments/watermark_inheritance/README.md#submitted-results) |
| Appendix D.1 Table 4, three Dolly epochs per descendant and the three final controls. | [Training](../experiments/watermark_inheritance/train.py), [archived run configurations](../experiments/watermark_inheritance/results/2026-10-06), [Dolly report](../experiments/watermark_inheritance/report.py). | Shared §5.1 witness draws and overlap-screened Dolly data. | [Dolly trajectory](../experiments/watermark_inheritance/results/2026-10-06/main_results.csv) |
| Appendix D.1 Table 5, matched UltraChat trajectory through 400,000 examples. | [Single-pass design](../experiments/watermark_inheritance/CURVE_DESIGN.md), [trajectory trainer](../experiments/watermark_inheritance/curve_train.py). | Original §5.1 parents and audit pairs, [deduplicated UltraChat preparation](../experiments/watermark_inheritance/curve_prepare.py). | [UltraChat trajectory](../experiments/watermark_inheritance/results/2026-10-06-distinct/curve.csv), [data manifest](../experiments/watermark_inheritance/results/2026-10-06-distinct/data_manifest.json) |
| §5.2.1, Table 2, Appendix D.2 Table 6, 14/14 public Pythia derivatives. | [Model and revision roster](../experiments/public_pythia_derivatives/runners/run_audit.py), [paper settings](../experiments/public_pythia_derivatives/configs/paper.yaml), [report compiler](../experiments/public_pythia_derivatives/reports/compile_results.py). | [Pinned MIMIR GitHub](../scripts/data/prepare_mimir_domain.py), 560/340 construction, 400 per source held out. | [Table 6](../experiments/public_pythia_derivatives/results/submitted_table_6.csv) |
| §5.2.1, Table 2, Appendix D.2 Table 7, 6/6 INT8 derivatives. | [Six-target launcher](../experiments/pythia_int8_quantization/jobs/run_paper_targets.sbatch), [runner and revision pins](../experiments/pythia_int8_quantization/runners/run_int8_robustness.py), [INT8 settings](../experiments/pythia_int8_quantization/configs/int8.yaml). | Same MIMIR GitHub preparation and splits as Table 6. | [Table 7](../experiments/pythia_int8_quantization/results/submitted_table_7.csv) |
| §5.2.2, Table 3, Appendix D.3 Table 8, 6/6 ViT derivatives, parent reference and random initialization. | [Checkpoint pins and launcher](../experiments/vit_provenance/jobs/run_public_derivatives.sbatch), [CLV](../experiments/vit_provenance/scoring/score_clv.py), [settings](../experiments/vit_provenance/configs/clv.yaml). | [Pinned ImageNet-1K class-balanced split](../experiments/vit_provenance/runners/prepare_shards.py), 1,925 per source per split. | [Table 8](../experiments/vit_provenance/results/submitted_table_8.csv) |
| §5.2.2, Table 3, Appendix D.4 Table 9, ViT merging, quantization and 30–90% pruning. | Shared CLV and threshold audit. The full checkpoint-construction recipe is unavailable. | Same ImageNet protocol as Table 8 with the published CIFAR-10 baseline. | [Table 9](../experiments/vit_provenance/results/submitted_table_9.csv) |
| Za6d witness reuse, 27 positive and 21 negative decisions at the adjusted cutoff 0.04347. | [Bonferroni check](../tests/test_rebuttal_results.py), using the §5.1 paired critical value with 48 tests. | 12 Dolly and 36 nonzero-exposure UltraChat audits from Tables 4–5. | [Checkpoint-level adjustments](../experiments/watermark_inheritance/results/2026-11-21-reviewer-analysis/submitted_48_multiplicity.csv) |
| qmGv Q1, two Pile-trained bases and their matched Lamini fine-tunes. | [Two-arm training recipe](../experiments/public_pythia_derivatives/configs/matched_lamini.yaml), [public-audit runner](../experiments/public_pythia_derivatives/runners/run_rebuttal_controls.py). | [Frozen MIMIR split hashes](../experiments/public_pythia_derivatives/configs/rebuttal_public_controls.json), 500 construction and 200 evaluation per source, plus both pinned Lamini Docs parquet files. | [Advantages and per-example numeric scores](../experiments/public_pythia_derivatives/results/rebuttal-public-controls/README.md) |
| qmGv Q3, public derivatives scored with MIN-K20 and mean log probability, 13/13 and 12/13 detections and 0/13 nonmember-control rejections for each. | [Shared scorer](../scripts/scoring/score_causal_lm_logprobs.py), [public runner](../experiments/public_pythia_derivatives/runners/run_audit.py), [compiler](../experiments/public_pythia_derivatives/reports/compile_results.py). | Matched MIMIR GitHub construction and evaluation on 340 and 400 per source. Exact matched input rows are unavailable. | [Observed comparison and rerun inputs](../experiments/public_pythia_derivatives/results/2026-11-23-aistats27-q3/README.md) |
| qmGv Q3 and oT8R Q3, BioGPT-Large (0.010, 0.000), Pythia step 0 (0.060, 0.045) and random ViT (0.0062). | [Pinned public-control models](../experiments/public_pythia_derivatives/configs/rebuttal_public_controls.json), [public-control runner](../experiments/public_pythia_derivatives/runners/run_rebuttal_controls.py), random ViT from Table 8. | Same frozen MIMIR split as qmGv Q1 for text. Same ImageNet protocol as Table 8 for ViT. | [Text results](../experiments/public_pythia_derivatives/results/rebuttal-public-controls/advantages.csv), [random ViT](../experiments/vit_provenance/results/submitted_table_8.csv) |
| qmGv Q3 and oT8R Q2/Q3, independently trained DeiT-Base at 0.0083. | [Pinned DeiT protocol](../experiments/vit_provenance/configs/aistats27_hard_negative.yaml), [hard-negative runner](../experiments/vit_provenance/runners/run_hard_negative.py). | Seed-0 class-balanced ImageNet preparation and the Table 8 CLV audit. | [DeiT result](../experiments/vit_provenance/results/2026-11-23-hard-negative/observed_summary.json) |
| oT8R Q2/Q4, 0/15 direct-base false alarms and the sample-size, α, and contamination grid. | [Control launcher](../experiments/watermark_inheritance/submit_aistats27_controls.sh), [model/shard seeds](../experiments/watermark_inheritance/configs/aistats27_false_positive.json), [calibration runner](../experiments/watermark_inheritance/calibration_analysis.py), [grid configuration](../experiments/watermark_inheritance/configs/aistats27_q4_calibration.json). | §5.1 generation and downstream preparation with independent seeds per control. | [Endpoints, per-unit grid, and aggregate calibration](../experiments/watermark_inheritance/results/2026-11-23-reviewer-q4-calibration/README.md) |
| oT8R Q5, theoretical bounds and Monte Carlo estimates at \(m=400,1600,8000\). | [Simulation](../experiments/power_simulation/reproduce.py), [fixed seed and sample sizes](../experiments/power_simulation/config.json). | Independent binomial draws on two shards. | [Full-precision values](../experiments/power_simulation/results.csv) |
| m7VV Q4, descendants, direct witness access and an independent same-key shard. | [Nine-epoch protocol](../experiments/watermark_inheritance/configs/m7vv_witness_sharing.json), [three-arm training](../experiments/watermark_inheritance/runners/m7vv_confirm_targets.py), [audit](../experiments/watermark_inheritance/matched_process_audit.py). | [Exact WikiText prompts, completion sampling, and 512 Dolly rows](../experiments/watermark_inheritance/runners/m7vv_confirm_shards.py). | [Audit, numeric scores and lineage hashes](../experiments/watermark_inheritance/results/m7vv-witness-sharing/README.md) |

## Reproduction limits

- **Watermark paper trajectories and 15-model calibration.** The submitted
  tables and archived per-model calibration counts support decision checks
  and exact aggregation. The original raw per-example score caches and
  checkpoints were unavailable in the publication environment. New GPU runs
  can regenerate scores using the pinned inputs and recipes.
- **Public Pythia and INT8.** The submitted tables retain the reported
  numbers. New training-source downloads require gated MIMIR access.
  `nnheui/pythia-1.4b-sft-full` is unavailable. The alternative-scorer result
  survives as an aggregate. Its original matched split, nonmember-control
  inputs and per-example outputs were unavailable for independent replay.
- **Matched Lamini and text reference controls.** The frozen numeric scores
  reproduce all 12 reported model–score values. The old MIMIR manifest lacks
  its source commit. The exact stored input bytes are identified by SHA-256.
  A new download from the pinned paper revision requires approved MIMIR
  access and has yet to be compared with those bytes.
- **Public ViT and transformations.** The submitted tables give the original
  values. An archived rerun using the available pinned reconstruction did
  not recover Table 8. For example the CIFAR-10 derivative reached 0.0291
  compared with the submitted 0.1138. Exact original image and run artifacts
  are needed to reconcile it. The original Table 9 merge inputs, complete
  transformation parameters and transformed checkpoints are unavailable.
  The saved DeiT summary supports its decision check. The original
  per-image DeiT output was unavailable.
- **Pythia-160M witness sharing.** Prompt and Dolly preparation match the
  original file hashes. The retained numeric scores reproduce the decisions.
  A fresh training and inference run requires a GPU and can differ numerically
  across GPU and library builds.
