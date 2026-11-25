# Reproducing the tested Pythia-160M recipe

Both independent replications used EleutherAI/pythia-160m at `50f5173d932e8e61f858120bcb800b97af589f46` and the original reading-mode watermark audit. Shared inputs reused from the earlier run include the generator/base weights and tokenizer, the pinned WikiText and Maryland code, and the exact 512 prepared Dolly rows. Six fresh parent checkpoints were necessary because each replication used a separately realized witness and distinct training seeds for its three arms. The retained lineages point to both the pre- and post-Dolly weights.

| Replication | P/Q/R witness seeds | A/B/C Dolly seeds | Witness schedule |
|---|---|---|---|
| One epoch | 61412 / 72632 / 83872 | 61513 / 72733 / 83973 | First 63 updates of one 189-update warmup/cosine cycle |
| Nine epochs | 91412 / 102632 / 113872 | 91513 / 102733 / 113973 | Three consecutive 189-update warmup/cosine cycles, carrying AdamW state |

Use a peak witness learning rate of 5e-5. Each cycle includes six warmup updates. Each epoch processes all 2,000 witness examples once in 63 optimizer updates. The effective batch is 32, microbatch eight. The trainer uses FP32 master parameters, BF16 autocast, full-parameter AdamW, weight decay 0.01 and gradient clipping at 1.0. It masks the prompt and pads, retaining the original completion IDs for causal supervision. Dolly uses a fresh optimizer, peak LR 1e-5, 16 updates and all 512 responses for one epoch. Supervise only S in the first stage of A and B, and S″ in the first stage of C. S′ is reserved for the audit.

Each replication has independent training draws and 2,000 articles, with disjoint article/prefix sets between the search, first confirmation and second confirmation. The key, prompts within each triple and generator distribution are shared between sides. Fit separate target/score thresholds by maximizing balanced accuracy on 400 construction pairs. Use the original strict comparison and largest tied threshold, then measure the signed S-minus-S′ difference on 1,600 evaluation pairs.

The exact training and audit programs are [minimal_q4_train.py](../../runners/minimal_q4_train.py), [m7vv_confirm_targets.py](../../runners/m7vv_confirm_targets.py), [m7vv_confirm_shards.py](../../runners/m7vv_confirm_shards.py), and [score.py](../../score.py). To reproduce a single branch on a separate output tree, run the following from the repository root after the nine-epoch data have been prepared. This example uses the nine-epoch A seed. B uses the same S projection with seed 102531 and C uses the S″ projection with seed 113771.

```bash
py=.cache/vit-venv/bin/python
wm=experiments.watermark_inheritance
fresh=artifacts/m7vv_minimal/confirmation_backup
dest=artifacts/m7vv_minimal_reproduction/A
for stage in parent target; do
    "$py" -m "$wm.runners.minimal_q4_train" bf16_lr5 "$stage" 9 \
        --root "$dest" --witness "$fresh/prepared/s_train.jsonl" --seed 91311 --save-only-requested
done
```

Stored artifacts are available from [one-epoch protocol](../../../../artifacts/m7vv_minimal/confirmation/protocol.json), [nine-epoch protocol](../../../../artifacts/m7vv_minimal/confirmation_backup/protocol.json), [one-epoch lineages](../../../../artifacts/m7vv_minimal/confirmation/lineages.json), [nine-epoch lineages](../../../../artifacts/m7vv_minimal/confirmation_backup/lineages.json), and the [time-stamped nine-epoch launch record](../../../../artifacts/m7vv_minimal/confirmation_backup/launch_context.json). A rerun's weight digest can differ despite matching data and seed, as observed in the original-recipe replay.
