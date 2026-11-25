"""Run pre-Dolly P/Q/R predictions as a secondary diagnostic after final prediction extraction."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.watermark_inheritance.common import write_json
from experiments.watermark_inheritance.score import file_digest
from experiments.watermark_inheritance.runners.m7vv_confirmation_cache import verify

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def run(root):
    if not (root/"timing.json").exists():
        raise RuntimeError("Final GPU prediction set not completed")
    selection=json.loads((root/"selected.json").read_text())
    fam,epoch=selection["family"],selection["witness_epochs"]
    dest=root/"stage_diagnostics"
    dest.mkdir(exist_ok=True,parents=True)
    for item in ("prepared","greenlist_cache"):
        link=dest/item
        if not link.exists():
            link.symlink_to((root/item).resolve(),target_is_directory=True)
    shutil.copyfile(root/"config.json",dest/"config.json")
    outputs=[]
    timing={}
    for target,parent in zip("ABC","PQR"):
        ck=root/"train"/target/fam/"checkpoints"/f"parent_epoch{epoch}"
        stage=ck/"stage_state.json"
        ledger=json.loads((root/"train"/target/fam/f"ledger_parent_epoch{epoch}.json").read_text())
        if not stage.exists() or ledger["weights_sha256"]!=file_digest(ck/"model.safetensors"):
            raise ValueError(f"Stage metadata and weights disagree at {ck}")
        start=timestamp()
        t=time.monotonic()
        with (dest/f"{parent}_prediction.log").open("w") as log:
            subprocess.run([sys.executable,"-m","experiments.watermark_inheritance.score",
                  "--run-dir",str(dest),"--checkpoint-name",parent,"--checkpoint-path",str(ck),"--predictions-only"],
                  stdout=log,stderr=subprocess.STDOUT,check=True)
        pred_seconds=time.monotonic()-t
        verify(dest,parent)
        timing[parent]=dict(start=start,end=timestamp(),gpu_prediction_seconds=pred_seconds,total_seconds=time.monotonic()-t,
                            source_sha256=ledger["weights_sha256"],checkpoint=str(ck))
        metrics=json.loads((dest/"fast_replay"/parent/"summary.json").read_text())
        for name,stat in metrics.items():
            ev=stat["evaluation"]
            outputs.append(dict(parent=parent,corresponding_target=target,witness_epochs=epoch,
                                score=name,fitted_threshold=stat["fit"]["threshold"],construction_ba=stat["fit"]["balanced_accuracy"],
                                accepted_S=stat["accepted_S"],accepted_S_prime=stat["accepted_S_prime"],gap=stat["precise_signed_gap"],
                                cutoff=stat["cutoff"],rejects=stat["rejects"],paired_p=ev["p_shard"]))
    with (dest/"pre_dolly.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=outputs[0]);w.writeheader();w.writerows(outputs)
    write_json(dest/"timing.json",timing)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",required=True,type=Path)
    run(ap.parse_args().run)
