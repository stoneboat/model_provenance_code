"""Assemble the frozen per-target decisions and secondary stage diagnostics for both replications."""
import csv
import json
import math
from pathlib import Path

from experiments.watermark_inheritance.calibration_analysis import paired_cutoff

ROOT = Path("artifacts/m7vv_minimal")
REPORT = Path("experiments/watermark_inheritance/results/2026-11-25-m7vv-minimal")

def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def main():
    result = []
    for epochs, run_name in [(1, "confirmation"), (9, "confirmation_backup")]:
        run = ROOT / run_name
        final = table(run / "audit/audit.csv")
        stages = [(run / "stage_diagnostics", "before_Dolly", list("PQR")), (run, "after_Dolly", list("ABC"))]
        for source, stage, labels in stages:
            for label, arm in zip(labels,"ABC"):
                scores = json.loads((source / "fast_replay" / label / "summary.json").read_text())
                for metric in ["watermark","min_k20"]:
                    row = scores[metric]
                    gap = (row["accepted_S"]-row["accepted_S_prime"])/1600
                    assert math.isclose(gap,row["evaluation"]["signed_gap"],abs_tol=1e-14)
                    primary = stage=="after_Dolly" and metric=="watermark"
                    six_cutoff = paired_cutoff(0.05/6,1600)
                    if stage=="after_Dolly":
                        official = next(r for r in final if r["target"]==label and r["detector"]==metric)
                        assert float(official["signed_advantage"])==gap
                        assert float(official["score_threshold"])==row["fit"]["threshold"]
                        assert float(official["exact_swap_p"])==row["evaluation"]["p_shard"]
                    result.append(dict(witness_epochs=epochs,stage=stage,checkpoint=label,final_arm=arm,score=metric,
                                       primary=primary,construction_pairs=400,heldout_pairs=1600,
                                       fitted_threshold=row["fit"]["threshold"],construction_ba=row["fit"]["balanced_accuracy"],
                                       accepted_S=row["accepted_S"],accepted_S_prime=row["accepted_S_prime"],
                                       signed_gap=gap,absolute_gap=abs(gap),paired_cutoff=row["cutoff"],
                                       rejected=abs(gap)>row["cutoff"],two_sided_swap_p=row["evaluation"]["p_shard"],
                                       six_primary_cutoff=six_cutoff if primary else "",
                                       rejected_six_primary=abs(gap)>six_cutoff if primary else ""))
    assert len(result)==24
    with (REPORT/"all_confirmation_outcomes.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=result[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)

if __name__ == "__main__":
    main()
