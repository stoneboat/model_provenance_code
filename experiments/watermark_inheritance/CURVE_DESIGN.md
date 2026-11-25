# Continuous clean fine-tuning trajectory

This extension measures how inherited watermark advantage changes with clean
fine-tuning exposure. It reuses the three completed private parents and their
matched base-model controls from the primary watermark experiment.

## Submitted single-pass design

The submitted Appendix D.1 experiment uses 600,000 distinct first-turn examples
from the full `openbmb/UltraChat` training split, pinned to revision
`f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0`. Each example is seen at most once.
The published paper reports the complete matched target/control audits available
through 400,000 examples.

Preparation applies the following fixed rules:

- retain the first user/assistant exchange;
- deduplicate normalized instructions and encoded training examples;
- reject exact and normalized overlap with `S` or `S′` from any of the three
  watermark runs; and
- use the same response-only, maximum-length-1,024 formatting as the Dolly
  experiment.

Training keeps the original effective batch size of 32 and uses one cosine
schedule over the full single pass. Checkpoints are fixed at optimizer steps
512, 1,024, 2,048, 3,125, 6,250, 12,500, and 18,750, corresponding to 16,384,
32,768, 65,536, 100,000, 200,000, 400,000, and 600,000 examples.

Each seed remains on one GPU and alternates its provenance target and matched
base-model control at checkpoint boundaries. Resumption restores the model,
optimizer, scheduler, random-number state, and data position. A fourth GPU
extracts audit predictions from atomically published checkpoints while CPU jobs
score them and refresh the trajectory reports.

Every checkpoint uses the original 2,000 calibration pairs, 8,000 held-out
audit pairs, watermark keys, wrong-key diagnostics, 100,000-token radioactivity
budget, and MIN-K baseline. Thresholds are recalibrated from the construction
split for each checkpoint and fixed before evaluation. Intermediate points are
descriptive repeated measurements; the pipeline does not claim that they are
independent hypothesis tests.

## Reproducibility safeguards

- The original parent config and paired-shard hashes are checked before reuse.
- The clean-data manifest records source revision, selection rules, overlap
  checks, token counts, and content hashes.
- Checkpoints are written atomically and only complete checkpoints can be
  scored.
- The tiny-model integration test verifies that interrupted and uninterrupted
  trajectories produce identical weights and exposure accounting.
