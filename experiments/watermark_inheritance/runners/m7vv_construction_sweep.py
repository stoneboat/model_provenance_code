"""Evaluate the prespecified epoch-by-family grid exclusively on original construction data."""
import argparse
import csv
from pathlib import Path
from time import monotonic

import torch
from transformers import AutoTokenizer
from experiments.watermark_inheritance.common import load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.runners.m7vv_construction import ORIGINAL, OUT, apply_membership, predict, summarize
from experiments.watermark_inheritance.runners.minimal_q4_train import FAMILIES

def run(epochs, stage, stronger=False):
    torch.set_num_threads(1)
    rows=read_jsonl(ORIGINAL/"prepared/pairs.jsonl")
    tokenizer=AutoTokenizer.from_pretrained(load_config(ORIGINAL)["tokenizer_path"],local_files_only=True)
    search=OUT.parent/"search"
    table=[]
    families=("bf16_lr5","head32_lr5") if stronger else FAMILIES
    screen="stronger" if stronger else "original"
    for epoch in epochs:
        for family in families:
            label=f"{family}_e{epoch}_{stage}"
            dest=OUT/label
            family_run=search/family
            if stage=="dolly":
                ck=family_run/f"dolly_epoch{epoch}"/"checkpoints/target_epoch1"
            else:
                ck=family_run/"checkpoints"/("parent" if epoch==24 else f"parent_epoch{epoch}")
            tick=monotonic()
            pred=predict(rows,tokenizer,ck)
            scored=apply_membership(rows,pred)
            metrics=summarize(scored)
            primary=metrics["z"]
            gaps=[fold["gap"] for fold in primary["crossfit"]]
            if stronger:
                gate=primary["twofold_gap"]>=0.20-1e-12 and min(gaps)>=0.15-1e-12
            else:
                gate=primary["twofold_gap"]>=0.12-1e-12 and min(gaps)>1e-12
            dest.mkdir(exist_ok=True,parents=True)
            write_jsonl(dest/"scores.jsonl",scored)
            write_json(dest/"summary.json",dict(label=label,checkpoint=str(ck),construction_pairs=len(scored),
                metrics=metrics,screen=screen,passes_screen=bool(gate),wall_seconds=monotonic()-tick))
            table.append(dict(epochs=epoch,family=family,stage=stage,keyed_crossfit_gap=primary["twofold_gap"],
                first_validation_gap=gaps[0],second_validation_gap=gaps[1],passes_screen=bool(gate),
                keyed_apparent_gap=primary["apparent_gap"],keyed_auc=primary["auc"],raw_mean_S=primary["mean_s"],
                raw_mean_Sprime=primary["mean_s_prime"],full_construction_threshold=primary["fit"]["threshold"],
                min_k20_crossfit_gap=metrics["min_k20"]["twofold_gap"],
                min_k20_auc=metrics["min_k20"]["auc"]))
            prefix="stronger_" if stronger else ""
            path=OUT/f"{prefix}sweep_{stage}_{epochs[0]}_{epochs[-1]}.csv"
            tmp=path.with_suffix(".tmp")
            with tmp.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
            tmp.replace(path)
            print(f"{label} keyed crossfit {primary['twofold_gap']:+.4f}; passed={gate}",flush=True)
            if stronger and gate:
                return

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("dolly","parent"),default="dolly")
    ap.add_argument("--stronger",action="store_true",help="Use the separate 0.20 pooled / 0.15 each-fold gate and stop at its first crossing")
    ap.add_argument("epochs",nargs="+",type=int)
    args=ap.parse_args()
    run(args.epochs,args.stage,args.stronger)
