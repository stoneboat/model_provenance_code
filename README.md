# Causal Model Provenance via Shard Witness Auditing

Code and result tables for the submitted AISTATS 2027 paper and its author
responses. The controlled experiments test what a witness audit can establish
about model ancestry. The public-model experiments measure detectable
training-data signals under their stated reference protocols.

## Experiments

| Package | Submitted paper | Author responses |
| --- | --- | --- |
| [Watermark inheritance](experiments/watermark_inheritance/README.md) | §5.1, Table 1, Appendix D.1 Tables 4–5 | Za6d witness reuse, oT8R Q2/Q4, m7VV Q4 |
| [Public Pythia derivatives](experiments/public_pythia_derivatives/README.md) | §5.2.1, Table 2, Appendix D.2 Table 6 | qmGv Q1/Q3 and oT8R Q3 |
| [Pythia INT8](experiments/pythia_int8_quantization/README.md) | §5.2.1, Table 2, Appendix D.2 Table 7 | |
| [ViT derivatives and transformations](experiments/vit_provenance/README.md) | §5.2.2, Table 3, Appendix D.3–D.4 Tables 8–9 | qmGv Q3 and oT8R Q2/Q3 |
| [Conditional-power simulation](experiments/power_simulation/README.md) | Proposition 6, §4.3 | oT8R Q5 |

The [experiment inventory](docs/EXPERIMENTS.md) maps each reported result to
its implementation, preparation, configuration, and saved evidence. It
records which results can be recalculated from the supplied evidence and
which reruns depend on original assets.

## Environment

Use Python 3.11 or 3.12. Create an isolated environment and install the
pinned dependencies, then work from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup/check_env.py
```

On PACE, use [the Slurm bootstrap](scripts/cluster/install_pace.sh) in place
of the virtual environment setup. Use a CUDA-capable PyTorch build compatible
with the GPU and driver. The watermark and Lamini training runs require BF16
support. INT8 inference uses bitsandbytes on a supported GPU. Slurm commands
accept site-specific account and partition overrides.

MIMIR and some model repositories require approved Hugging Face access.
Authenticate using the Hub client or provide `HF_TOKEN` in your environment.
Download model and dataset inputs from their pinned upstream repositories
under their respective access and license terms. Downloaded inputs, weights,
checkpoints, and raw scoring outputs are generated locally under `.cache/`
and `artifacts/`. The committed numeric evidence contains scores and
aggregate statistics.

## Verify and reproduce

```bash
python -m pytest -q tests experiments/watermark_inheritance/tests
python -m experiments.watermark_inheritance.runners.verify_m7vv
python -m experiments.public_pythia_derivatives.reports.verify_rebuttal_controls
python experiments/power_simulation/reproduce.py --check
find experiments scripts -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec bash -n {} +
```

The upstream watermark compatibility tests expect a source checkout at
`.cache/watermark/radioactive-watermark`. Set it up using
[the pinned watermark source instructions](experiments/watermark_inheritance/DESIGN.md).
Each experiment README gives the model and data preparation commands for
new GPU runs. Shared preparation, scoring and auditing commands are under
[`scripts/`](scripts/README.md), with reusable Python code under
[`src/shard_audit/`](src/shard_audit/README.md).
