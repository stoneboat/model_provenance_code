# Alternative scores on public Pythia derivatives

For qmGv Q3, both distinguishers used the same checkpoints and cached
next-token log probabilities. MIN-K20 averages the lowest 20% of eligible
token log probabilities. The second distinguisher averages all of them.
Thresholds maximize balanced accuracy on the construction split. The
reported matched run used 340 construction and 400 held-out examples per
source from the pinned MIMIR GitHub split.

| Score | Witness detections | Nonmember-control rejections |
| --- | ---: | ---: |
| MIN-K20 | 13/13 | 0/13 |
| Mean log probability | 12/13 | 0/13 |

`kykim0/pythia-1.4b-tulu-v2-mix` was detected only by MIN-K20. The
fourteenth paper checkpoint, `nnheui/pythia-1.4b-sft-full`, was unavailable.
[Observed summary](observed_scorer_summary.json) · [Matched split provenance](matched_split.json)

The exact matched inputs and original model-level score files are needed for
a numerical rerun. Given those inputs, place the original prepared directories
under `artifacts/public_pythia_derivatives/aistats27_matched/seed_0/prepared/`
using the names `mimir_github` and `mimir_github_nonmember_control`.
Then use the shared model roster, scorer, and compiler.

```bash
root=artifacts/public_pythia_derivatives/aistats27_matched
python experiments/public_pythia_derivatives/runners/run_audit.py \
  --artifact-root "$root" --skip-prepare --skip-unavailable --resume
python experiments/public_pythia_derivatives/reports/compile_results.py --artifact-root "$root"
```
