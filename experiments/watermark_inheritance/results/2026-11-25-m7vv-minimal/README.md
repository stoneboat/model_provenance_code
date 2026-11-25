# Nine witness epochs produced an independently verified watermark result

The nine-epoch Pythia-160M replication met the prespecified three-arm outcome. The top-1 keyed watermark audit detected the genuine descendant A and the independent same-witness target B. The fresh same-key target C stayed below the detection cutoff. The independent one-epoch replication failed the primary endpoint, although MIN-K20 detected A and B in both replications.

## Held-out signed advantages

Each entry gives the difference in acceptance rates, S minus S′, before and after Dolly. Each threshold was fitted on its own checkpoint's 400 construction pairs and applied to 1,600 held-out pairs from the same replication. The post-Dolly watermark decisions are primary. MIN-K20 and all pre-Dolly measurements are secondary. The absolute paired cutoff was 0.067905 at per-target α = 0.05.

| Witness epochs | Development history | Watermark before → after Dolly | Primary final decision | MIN-K20 before → after Dolly | MIN-K20 final decision |
|---:|---|---:|---|---:|---|
| 1 | P → A, genuine descendant | +0.001250 → +0.011875 | No | +0.241875 → +0.188750 | Yes |
| 1 | Q → B, independent on S | +0.006250 → +0.015625 | No | +0.296875 → +0.245625 | Yes |
| 1 | R → C, independent on S″ | +0.000625 → 0.000000 | No | −0.010000 → +0.001250 | No |
| 9 | P → A, genuine descendant | +0.133125 → **+0.136875** | **Yes** | +0.679375 → +0.664375 | Yes |
| 9 | Q → B, independent on S | +0.139375 → **+0.096875** | **Yes** | +0.670000 → +0.642500 | Yes |
| 9 | R → C, independent on S″ | −0.008750 → **+0.003750** | **No** | +0.010000 → +0.002500 | No |

For the nine-epoch final targets, the keyed audit accepted **820 vs 601** S/S′ completions for A (two-sided paired-swap p = 1.82 × 10⁻¹⁴), **998 vs 843** for B (p = 1.43 × 10⁻⁸), and **277 vs 271** for C (p = 0.814). Both positive gaps also exceed **0.082770**, the conservative Bonferroni paired cutoff over all six final-target primary tests in the two replications. For the one-epoch final targets, the corresponding p-values were 0.324, 0.0967 and 1.000. The independent C non-rejections are individual observations; they cannot estimate a population false-positive rate. [All score thresholds, balanced accuracies, acceptance counts, gaps, and exact p-values](all_confirmation_outcomes.csv)

## Why the result matters for m7VV Q4

S is an input to the designated P, and A obtains that information through P. B starts again from the shared pretrained base and has direct access to exactly the same realized S. This bypass violates exclusion for the named parent. C receives S″, which is independently sampled conditional on the same prompts, generator, key, and decoding law; its access to the random realization of S has no such route. Thus B's positive result demonstrates shared-witness dependence in a known non-descendant. The independent nine-epoch replication also directly localizes an observable signal at P and its persistence through the fixed Dolly stage.

The earlier three-epoch LR 1e-5 parent was discarded. Replaying its original inputs, seed, optimizer settings and step count gave a different weight digest. In the *matched replay chain*, the keyed gap was +0.025625 before and +0.025000 after Dolly, both below cutoff; MIN-K20 gaps were +0.159375 and +0.155000, both detected. The fresh one-epoch parent was likewise already weak before Dolly. These observations support weak initial keyed learning in those runs. Because the one- and nine-epoch confirmations use different draws and seeds, their difference does not identify a pure epoch effect. The exact minimum sufficient training duration remains open. [Retrospective replay](replay_retrospective.csv) · [Archived reproduction check](../../../../artifacts/m7vv_minimal/replay/verification.json)

## Selection, independent data and cost

The first search evaluated three witness LR/precision recipes at 1, 2, 3 and 4 epochs using only the original construction pairs. The chosen one-epoch BF16 LR 5e-5 recipe had pooled two-fold construction gap +0.1275 (folds +0.055, +0.200). A separate higher-exposure construction search selected nine epochs at the same LR, with pooled gap +0.2225 (folds +0.225, +0.220). Both six-epoch recipes missed that search's stricter gate. The two confirmation recipes and their distinct sampling and training seeds were fixed before any fresh outcome was available. At the user's request, nine-epoch confirmation began unconditionally while the first confirmation dataset was still being generated. Both final outcomes are reported, and both confirmation datasets are disjoint from the development data and from each other. [Twelve short candidates](results.csv) · [Higher-exposure candidates](backup_screen.csv) · [Time-stamped launch](../../../../artifacts/m7vv_minimal/confirmation_backup/launch_context.json) · [Integrity checks](../../../../artifacts/m7vv_minimal/confirmation_integrity.json)

All confirmation work used one A100-SXM4-40GB. The nine-epoch run reused the pinned Pythia base and tokenizer, the sampler, the exact 512 Dolly rows and the existing training/scoring implementation. Each new model had to train on its assigned fresh realization. Generation took about 26m25s. Training all three fresh branches through 567 witness updates and 16 Dolly updates each, saving their weights and extracting the final predictions took another 6m49s. The final GPU predictions were saved at 01:46:05 UTC. P/Q/R and A/B/C are retained for both replications. [Exact setup, seeds and reproduction commands](reproduction.md) · [One-epoch checkpoints](../../../../artifacts/m7vv_minimal/confirmation/lineages.json) · [Nine-epoch checkpoints](../../../../artifacts/m7vv_minimal/confirmation_backup/lineages.json) · [Final nine-epoch timing](../../../../artifacts/m7vv_minimal/confirmation_backup/timing.json)

## Rebuttal-ready paragraph

We compared three independently documented Pythia-160M development histories on fresh paired watermarked completions. After nine witness-training epochs and identical Dolly fine-tuning, the primary watermark audit gave signed advantages of 0.1369 for the genuine descendant, 0.0969 for a model trained independently on the same realized witness, and 0.0038 for a model trained on an independent same-key shard. The first two exceeded the paired cutoff of 0.0679 and remained positive under correction across all six primary tests in our one- and nine-epoch replications. The one-epoch replication detected none of the three with the primary score. These results demonstrate that detectable witness dependence can persist through downstream fine-tuning and can also arise in a non-descendant with direct witness access. Parent attribution therefore requires excluding that alternative route.
