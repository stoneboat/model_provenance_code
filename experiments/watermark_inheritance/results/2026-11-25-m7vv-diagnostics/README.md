# Why the 160M watermark audit missed the genuine descendant

**The final A checkpoint assigns relatively higher probabilities to the least likely witness tokens, but its top-ranked tokens are about equally likely to be green on S and S′.** The original threshold reproduced exactly. Across every possible threshold on the saved watermark z-scores, the largest absolute held-out acceptance gap for A is 0.024375. The prespecified rejection cutoff was 0.067905. Changing the threshold could not have changed this decision.

This is a retrospective diagnostic of one frozen experiment. We reused all S/S′ scores and ran inference on existing final checkpoints for S″. New precision and probability probes used construction data only. P, Q and R are the first-stage checkpoints leading to A, B and C. Only their tokenizer, configuration, digests and training records remain. Neither their weight files nor pre-Dolly scores are present in the run or evidence archive. Consequently a pre-/post-Dolly advantage comparison for the 160M models is unavailable. [Frozen results](../2026-11-24-m7vv-q4/README.md) · [Checkpoint retention record](../../../../artifacts/m7vv_q4/train/training_summary.json) · [Byte-for-byte preservation checks](preservation.json)

## What each score saw

All z summaries are mean ± sample SD on the same 1,600 evaluation prompts. AUC and paired differences compare S with S′, even for C, whose training shard was S″. Threshold and balanced accuracy (BA) come from the original 400 construction pairs.

| Final target | S z | S′ z | S″ z | Paired mean z difference [95% interval] | Held-out AUC | Fitted z threshold | Construction BA | Frozen watermark / MIN-K20 signed advantage |
|---|---|---|---|---|---:|---:|---:|---:|
| A, through P(S) | 0.287 ± 1.000 | 0.262 ± 1.017 | 0.310 ± 1.009 | +0.025 [−0.043, +0.094] | 0.505 | 0.000113 | 0.54375 | −0.005625 / +0.108750 |
| B, through Q(S) | 0.371 ± 0.999 | 0.349 ± 1.025 | 0.374 ± 1.015 | +0.022 [−0.047, +0.090] | 0.502 | 1.393686 | 0.53500 | −0.007500 / +0.167500 |
| C, through R(S″) | 0.312 ± 1.016 | 0.334 ± 0.994 | 0.394 ± 0.999 | −0.022 [−0.091, +0.046] | 0.493 | −0.387324 | 0.54625 | −0.007500 / −0.010625 |

The empirical CDF distance bounds the absolute acceptance gap at every threshold without selecting a replacement. For A, B and C the distances on S versus S′ were 0.0244, 0.0369 and 0.0231. Even the largest distance across all three models and all shard pairings was only 0.0456. The higher construction BA of C than A also cautions against reading much into a fitted construction accuracy around 54%.

MIN-K20 uses the probabilities assigned to the *observed* least likely fifth of completion tokens. Its S-versus-S′ evaluation AUCs were 0.588 for A, 0.617 for B and 0.498 for C. For C's own training shard S″ versus S′, MIN-K20 AUC was 0.614 and the gap at the original S-versus-S′ threshold was +0.1444. The keyed watermark AUC on that same contrast was 0.518 and the transferred gap +0.0300. These S″ comparisons are diagnostic. [Distributions, both splits](score_distributions.csv) · [All paired comparisons, fits and thresholds](paired_score_comparisons.csv) · [Empirical CDF figure](heldout_ecdfs.png) · [Statistical methods](distribution_methods.md)

## Generation, scoring and numerical precision

The eligible **generated tokens** were green on 80.35% of S, 80.29% of S′ and 80.52% of S″ under key 101. The corresponding wrong-key fractions were 25.15%, 25.15% and 25.05%. The primary audit instead reads the **target's unbiased top-1 prediction**, scores it against the greenlist, and standardizes the green count on nonrepeated eligible contexts. We checked the two-token context, salt 35317, greenlist size 12,569 of 50,277 valid tokens, tokenizer-ID maps, exclusion of 27 padded IDs, completion-only masks and one-token causal shift. An independent implementation reproduced all 12,000 saved A/B/C S/S′ per-example reading scores. Fresh inference also reproduced a complete original batch for each target. [Key and alignment checks](key_and_alignment.md) · [Actual emitted-token scores](emitted_watermark.csv)

A real numerical limitation compounds the top-1 readout. On a fixed 7,989-position construction probe, final A had median peak logit 832, where BF16 values are spaced four units apart. 69.56% of positions had multiple winning tokens in BF16; 0.013% did in FP32. The pretrained base already had this behavior. We then compared inference precision on **all 400 construction pairs**, holding batch size and context constant. Watermark AUCs were 0.5243 with BF16, 0.5280 with a full-precision output projection on BF16 hidden states, and 0.5242 with full FP32. Removing ties did little for shard discrimination on construction. These diagnostic forwards used batches of eight and the original audit used 32 completions per batch. BF16 top-1 agreement with the cached outputs was about 90% after rebatching; the original batch replay agreed exactly. The archived construction AUC was 0.5419. [Precision comparison](construction_precision_probe.json) · [Base and final precision checks](key_and_alignment.md)

A separate construction-only probe compared the pretrained base and final A. The mean MIN-K20 S-minus-S′ difference shifted from −0.0437 to +0.1728. For final A the fraction of positions predicting the exact observed next token was 21.47% on S and 21.50% on S′. Mean greenlist probability mass was 26.11% versus 25.58%, measured at up to eight evenly spaced eligible contexts per example. Absolute mean token log probability worsened from base to final A on both shards in BF16. The MIN-K result concerns a *relative* witness advantage. [Construction probability probe](construction_probability_probe.json)

## Comparison with the successful 1B run

| Setting | Pythia-160M, A | Original Pythia-1B, run 00 |
|---|---:|---:|
| Model used for generation and training | 160M | 1B |
| Witness rows × epochs | 2,000 × 3 | 10,000 × 3 |
| Witness optimizer updates | 189 | 939 |
| Prompt length / completion cap | 64 / 256 | 64 / 256 |
| Observed mean completion length on S and S′ | 236.0 | 239.1 |
| AdamW peak LR / effective batch | 1e−5 / 32 | 1e−5 / 32 |
| Dolly rows × epochs | 512 × 1 | 15,011 × 3 |
| Dolly optimizer updates | 16 | 1,410 |
| Final witness-epoch online training NLL | 4.157 | 1.514 |

The shared key, bias, sampling temperature, top-p, witness epochs and nominal learning rate match. The newer run used a 512-token training cap versus 1,024 originally, but every generated example fit in both. The runs also differed in generator, microbatch and software environment. The 1B keyed advantage on the matching key was +0.67275 at P, +0.492375 after its first full Dolly epoch and +0.590625 after the third. The corresponding MIN-K20 advantages were +0.7315, +0.558875 and +0.736375. The 1B parent already had strong separation before Dolly; its correct-key aggregate top-1 green rates were 37.23% on S and 27.45% on S′. Online training losses were measured on different generators' completions and do not isolate capacity. [Configuration and exposure table](training_comparison.csv) · [Original results](../2026-10-06/results.json) · [Original training records](../2026-10-06/training_validation.json)

## Interpretation and cheapest next experiment

The observed failure is in final-target top-1 greenlist separation. The generated watermark was strong, the scorer used the right key and positions, the frozen probabilities retained information about exposure, and no scalar z threshold could separate the observed evaluation scores enough. Lower capacity and approximately fivefold fewer witness optimizer updates are plausible contributors to weaker acquisition. BF16 ties demonstrably make the reading unstable, but a precision-only change did not substantially improve construction AUC. Differences in capacity, exposure, generator and precision during training cannot be separated here. The 160M model received vastly less Dolly training than the successful run, and a loss of signal during those 16 updates remains possible.

The cheapest experiment to distinguish weak acquisition from Dolly erasure is to replay **only A's witness stage**, unchanged, with seed 34812, the pinned base and existing S. That stage previously took 38 seconds on the A100, excluding scoring. Save P, compare its weight digest to the archived digest, and audit its 400 construction pairs alongside saved A using the original scorer and batching. A strong P signal followed by weak A would implicate the downstream stage; weak scores at both would point to acquisition or readout. If the digest differs, label P a replication. For a confirmatory change to training or readout, freeze the change and use a fresh independent audit realization. No training or held-out tuning was performed for these diagnostics.
