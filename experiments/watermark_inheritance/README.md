# Watermark-based LLM provenance

This package is the implementation and completed result bundle for Section 5.1
of the submitted paper:

- main-paper Table 1: final Dolly and UltraChat provenance decisions;
- Appendix D.1, Table 4: the full three-epoch Dolly trajectories; and
- Appendix D.1, Table 5: the UltraChat trajectories through 400,000 distinct
  examples.

## Experiment

For each of three seeds and watermark keys, the pipeline creates 10,000 matched
WikiText completion pairs under the same Maryland/Kirchenbauer watermark. One
randomly assigned completion per prompt enters the secret shard `S`; the other
enters the paired control `S′` and is never used for training.

```text
B = EleutherAI/pythia-1b

B -- full fine-tuning on S --> P -- clean fine-tuning --> provenance target T
B ---------------------------- clean fine-tuning --> no-provenance control T0
```

The reading-mode distinguisher is calibrated on 2,000 pairs and evaluated on
8,000 different pairs. The paired level-0.05 cutoff is 0.0304. The pipeline
also records the base model, the private parent, every Dolly epoch, wrong-key
radioactivity checks, and MIN-K 20% diagnostics. See [DESIGN.md](DESIGN.md) for
the complete protocol.

## Submitted results

The final signed shard advantages reproduce the submitted Table 1:

| Downstream data | Run | Provenance target | No-provenance control |
|---|---:|---:|---:|
| Dolly | 00 | 0.5906 | 0.0013 |
| Dolly | 01 | 0.6020 | -0.0083 |
| Dolly | 02 | 0.6160 | -0.0035 |
| UltraChat, 400k examples | 00 | 0.3124 | 0.0003 |
| UltraChat, 400k examples | 01 | 0.3150 | -0.0078 |
| UltraChat, 400k examples | 02 | 0.3228 | -0.0041 |

All six provenance targets reject the null and all six controls fail to reject,
matching the submitted paper. Published artifacts are versioned with the code:

- [Dolly paper rows](results/2026-10-06/main_results.csv)
- [UltraChat trajectory rows](results/2026-10-06-distinct/curve.csv)

## Reproduce the Dolly experiment

Install the repository environment first, then submit the full dependency chain
from the repository root:

```bash
bash experiments/watermark_inheritance/submit.sh
```

The setup job creates all three immutable configs, pins and downloads the model,
datasets, and upstream watermark implementation, runs the unit tests, and builds
the paired prompt inputs. The dependency chain then runs the GPU pilot, full
generation/training/inference, CPU scoring, and final report.

The defaults target this repository's PACE environment. These variables make
the Slurm submission portable without editing committed files:

| Variable | Default | Purpose |
|---|---|---|
| `WM_PYTHON` | `.venv_pace/bin/python` | Python executable |
| `WM_ACCOUNT` | scheduler default | CPU account |
| `WM_GPU_ACCOUNT` | `WM_ACCOUNT` | GPU account |
| `WM_CPU_PARTITION` | scheduler default | CPU partition |
| `WM_GPU_PARTITION` | scheduler default | GPU partition |
| `WM_GPU_TYPE` | `h200` | Slurm GPU resource type; set empty for generic GPUs |
| `WM_ARTIFACT_ROOT` | `artifacts/watermark_inheritance` | Run state and checkpoints |
| `WM_SHARED_DIR` | `.cache/watermark` | Pinned model/data cache |

Example with explicit PACE routing:

```bash
WM_ACCOUNT=<account> \
WM_GPU_PARTITION=<gpu-partition> \
bash experiments/watermark_inheritance/submit.sh
```

Generation is journaled and resumable, checkpoints are published atomically,
and existing artifacts are validated before reuse. Large generated shards and
model checkpoints stay under `artifacts/` and are intentionally ignored by Git.

## Reproduce the UltraChat appendix trajectory

The submitted UltraChat curve reuses the completed parents and audit pairs from
the Dolly pipeline. It trains matched target/control branches over 600,000
distinct first-turn examples in a single pass and reports the checkpoints
available through 400,000 examples in the paper:

```bash
bash experiments/watermark_inheritance/submit_ultrachat.sh
```

The source dataset is pinned to `openbmb/UltraChat` revision
`f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0`. Instructions and encoded examples
are deduplicated, and exact plus normalized overlap with all three private shards
is excluded. See [CURVE_DESIGN.md](CURVE_DESIGN.md) for the trajectory design.

## Verification

Run the complete static and tiny-model test suite with the project environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TRITON_CACHE_DIR=.cache/triton MPLCONFIGDIR=.cache/matplotlib \
  .venv_pace/bin/python -m unittest discover \
  -s experiments/watermark_inheritance/tests -v
```

The tests cover upstream watermark compatibility, added tokenizer tokens,
independent paired sampling, source/pilot separation, overlap filtering,
response masks, gradient accumulation, exact resume behavior, teacher-forcing
alignment, deduplication, calibrated thresholds, and paired inference. The GPU pilot separately validates generation quality,
parameter updates, checkpoint reloads, memory use, and the aggregate token
budget before any production job can run.

## Rebuttal experiments

- [Za6d witness reuse](results/2026-11-21-reviewer-analysis/README.md) applies
  Bonferroni correction to the 48 audits in Appendix Tables 4–5.
- [oT8R Q2 and Q4](results/2026-11-23-reviewer-q4-calibration/README.md) cover
  15 independently fine-tuned direct-base controls and the calibration grid.
- [m7VV Q4](results/m7vv-witness-sharing/README.md) tests three development
  histories on independently sampled Pythia-160M completions.
