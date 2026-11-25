# Appendix Tables 4–5 with family-wise error control

The Za6d witness-reuse response applies Bonferroni across the 48 reported
checkpoint-level audits. The family contains the 12 Dolly rows in Table 4
and the 36 nonzero-exposure UltraChat audits in Table 5. It excludes the
zero-exposure base and parent references.

At α = 0.05 and 8,000 evaluation pairs, the paired threshold increases
from 0.03037 to 0.04347. The 27 target detections and 21 control
non-rejections are the same under both thresholds. Bonferroni allows
dependence across repeated models, prompts, and witnesses.

[Checkpoint-level results](submitted_48_multiplicity.csv) · [Family summary and input hashes](submitted_48_multiplicity.json)

```bash
python -m pytest -q tests/test_rebuttal_results.py -k witness_reuse
```
