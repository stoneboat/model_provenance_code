"""Independently verify the stronger-dose screen from construction scores and fixed folds."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RUN=Path("artifacts/m7vv_minimal/backup")
OUT=Path("experiments/watermark_inheritance/results/2026-11-25-m7vv-minimal")
EPOCHS=(6,9,12,18,24)
FAMILIES=("bf16_lr5","head32_lr5")

def fit(x,y):
    candidates=np.unique(np.r_[x,y])
    candidates=np.r_[math.nextafter(candidates[0],-math.inf),candidates]
    n_correct=(x[:,None]>candidates).sum(0)+(y[:,None]<=candidates).sum(0)
    return float(candidates[np.flatnonzero(n_correct==n_correct.max())[-1]])

def main():
    folds=json.loads((OUT/"construction_folds.json").read_text())
    ids=set(folds["fold_0"]+folds["fold_1"])
    ordered=sorted(ids,key=lambda i:hashlib.sha256(("m7vv-split-v1:"+i).encode()).digest())
    assert folds["fold_0"]==ordered[::2] and folds["fold_1"]==ordered[1::2] and len(ids)==400
    all_results=[]
    ended=False
    for epoch in EPOCHS:
        for family in FAMILIES:
            d=RUN/"construction"/f"{family}_e{epoch}"
            if not (d/"summary.json").exists():
                ended=True
                continue
            assert not ended,"Observed candidate after a missing or passing lower-cost candidate"
            saved=json.loads((d/"summary.json").read_text())
            examples={r["pair_id"]:r for r in (json.loads(l) for l in (d/"scores.jsonl").open())}
            assert set(examples)==ids
            metrics={}
            passed=False
            for key in ("z","min_k20"):
                values=[np.array([[examples[p][side][key] for side in ("s","s_prime")] for p in folds[f"fold_{i}"]]) for i in [0,1]]
                gaps=[]
                for train,test in (values,values[::-1]):
                    t=fit(train[:,0],train[:,1])
                    gap=(int((test[:,0]>t).sum())-int((test[:,1]>t).sum()))/len(test)
                    gaps.append(gap)
                all_scores=np.concatenate(values)
                threshold=fit(all_scores[:,0],all_scores[:,1])
                assert threshold==saved["metrics"][key]["fit"]["threshold"]
                assert np.allclose(gaps,[r["gap"] for r in saved["metrics"][key]["crossfit"]],rtol=0,atol=1e-12)
                metrics[key]=gaps
            pooled=sum(metrics["z"])/2
            passed=pooled>=.20-1e-12 and min(metrics["z"])>=.15-1e-12
            assert passed==saved["passes_backup_gate"]
            all_results.append(dict(epoch=epoch,family=family,keyed_crossfit_gap=pooled,fold1=metrics["z"][0],
                        fold2=metrics["z"][1],passes=passed,min_k20_crossfit_gap=sum(metrics["min_k20"])/2))
            ended=passed
    if all_results:
        path=OUT/"backup_screen.csv"
        temp=path.with_suffix(".tmp")
        with temp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=all_results[0]);w.writeheader();w.writerows(all_results)
        temp.replace(path)
    print(all_results[-1] if all_results else "No completed backup candidates")

if __name__=="__main__":
    main()
