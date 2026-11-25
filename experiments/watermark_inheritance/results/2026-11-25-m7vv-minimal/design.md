# Pythia-160M witness-stage replay and minimal-setup search

## What counts as working

The primary endpoint is the original keyed top-1 reading score after the matched Dolly stage. A successful independent replication has positive held-out detections for the genuine descendant A and the independent same-S target B, with no absolute-gap rejection for the independent fresh-S″ target C. Use the original per-target two-sided cutoff sqrt(2 log(40)/1600) = 0.0679051. Report every target, its fitted threshold, signed and absolute gaps, acceptance counts and exact paired-swap p-value. MIN-K20 is secondary.

## Stage diagnosis

Replay the original A parent from the same pinned base and original training projection with original seed, hyperparameters and 189 optimizer updates. Compare the weight-file digest against the archived P digest. If it differs, train a matched Dolly target from the replay itself to compare pre/post scores within the same chain. Use the original primary readout and original construction/evaluation split for this retrospective comparison; its evaluation rows are excluded from recipe selection.

## Search fixed before candidate scores

Hold the base, original 2,000 S rows, Dolly data (512 rows for one epoch), and BF16 audit fixed. Search over witness training only. Order candidates first by completed witness epochs, then by the recipe order below. The unit is an epoch of 63 updates at effective batch 32, including the partial final batch. This searches minimum witness exposure in the stated grid, not minimum model size, data count, or wall time across all possible setups.

1. Original BF16 training, peak LR 0.00001.
2. Original BF16 training, peak LR 0.00005.
3. BF16 backbone with a full-precision output projection, peak LR 0.00005.

Check prefixes of 1, 2, 3, 4, 6, 9, 12, 18 and 24 witness epochs as needed. For each family, repeat the original three-epoch warmup/cosine LR schedule, carrying optimizer state across cycles. Train each prefix's Dolly target separately with the original downstream BF16 recipe and reset optimizer. Within the three-epoch LR cycle, a shorter prefix stops early; it does not rescale the schedule. Each search arm uses the same model seed and the same original rows.

Selection uses only the original 400 construction pairs. Deterministically split the 400 prompt pairs into two halves by sorting the SHA-256 digests of a fixed prefix plus pair ID and taking alternating rows. Fit the original strict threshold on one half and measure the signed acceptance gap on the other, then reverse. A post-Dolly candidate qualifies at a pooled cross-fitted gap of at least +0.12 with positive gaps in both halves. The margin above the 1,600-pair cutoff is a screening heuristic, not a significance claim. Finish all lower-cost candidates needed to establish the first qualified recipe. Do not choose using original or fresh held-out outcomes.

## Independent confirmation

Use new disjoint article prompts, the same generator, key, sampling law and 256-token cap, and independent S/S′/S″ draws; fix 400 construction and 1,600 evaluation pairs. Train fresh A and B on the new shared S and C on the new S″, then apply the same 512 Dolly examples to all. Keep S′ for auditing. Fix the recipe before new audit results; no retuning after confirmation. Each new model and score separately fits a threshold on the 400 new construction pairs. Releasing the old witness-only model or scoring a new S that the target never saw would not test witness retention. Retain final verification artifacts and only necessary weights.
