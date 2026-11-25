"""Replicate one frozen development recipe on the three independently generated confirmation arms."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.watermark_inheritance.common import load_json, write_json
from experiments.watermark_inheritance.prepare import file_sha256
from experiments.watermark_inheritance.runners.m7vv_confirm_shards import RUN
from experiments.watermark_inheritance.runners.minimal_q4_train import FAMILIES, PREFIXES


def now():
    return datetime.now(timezone.utc).isoformat()


def run(family,epoch,deadline,run_root=RUN):
    start=time.monotonic()
    def remaining():
        seconds=deadline-time.time()
        if seconds <= 0:
            raise TimeoutError("Confirmation GPU allocation expired")
        return seconds
    protocol=load_json(run_root/"protocol.json")
    frozen=load_json(run_root/"selection_before_generation.json")
    if (family,epoch)!=(frozen["family"],frozen["witness_epochs"]) or frozen["protocol_sha256"]!=file_sha256(run_root/"protocol.json"):
        raise ValueError("Selected recipe or independent protocol changed since the pre-generation freeze")
    data=load_json(run_root/"prepared/shards.manifest.json")
    config=dict(family=family,witness_epochs=epoch,protocol_sha256=file_sha256(run_root/"protocol.json"),
                confirmation_shards_manifest_sha256=file_sha256(run_root/"prepared/shards.manifest.json"))
    selection=run_root/"selected.json"
    if selection.exists() and load_json(selection)!=config:
        raise ValueError("Confirmation already bound to a different selected recipe or data")
    write_json(selection,config)
    audit=run_root/"audit"
    audit.mkdir(parents=True,exist_ok=True)
    children=[]
    models={}
    started=now()
    with ExitStack() as files:
        for arm in ("A","B","C"):
            branch=protocol["branches"][arm]
            armroot=run_root/"train"/arm
            armroot.mkdir(parents=True,exist_ok=True)
            witness_key="s_double_prime_train" if arm=="C" else "s_train"
            witness=run_root/"prepared"/(witness_key+".jsonl")
            if file_sha256(witness)!=data["files"][witness_key]:
                raise ValueError(f"Fresh training projection changed for {arm}")
            family_dir=armroot/family
            branch_dir=family_dir/f"dolly_epoch{epoch}"
            for stage in ("parent","target"):
                ledger=(family_dir if stage=="parent" else branch_dir)/f"ledger_{stage}_epoch{epoch}.json"
                if ledger.exists():
                    identity=load_json(ledger)
                    if identity["weights_sha256"]!=file_sha256(Path(identity["checkpoint"])/"model.safetensors"):
                        raise ValueError("Existing model differs from its training ledger")
                    continue
                cli=[sys.executable,"-u","-m","experiments.watermark_inheritance.runners.minimal_q4_train",family,stage,str(epoch),
                     "--root",str(armroot),"--seed",str(branch["model_seed"]),"--witness",str(witness),"--save-only-requested"]
                with (armroot/f"{stage}.log").open("a") as log:
                    subprocess.run(cli,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
            parent=load_json(family_dir/f"ledger_parent_epoch{epoch}.json")
            target=load_json(branch_dir/f"ledger_target_epoch{epoch}.json")
            if (parent["data_sha256"]!=data["files"][witness_key] or parent["weights_sha256"]!=target["source_sha256"]
                    or parent["updates"]!=63*epoch or target["updates"]!=16
                    or parent["seed"]!=branch["model_seed"] or target["seed"]!=branch["model_seed"]):
                raise ValueError(f"Unverified training chain for {arm}")
            intermediate_optimizer=Path(parent["checkpoint"])/"training_state.pt"
            if intermediate_optimizer.exists():
                intermediate_optimizer.unlink()
            models[arm]=dict(parent=parent,target=target)
            write_json(run_root/"lineages.json",models)
            args=[sys.executable,"-u","-m","experiments.watermark_inheritance.score","--run-dir",str(run_root),
                  "--checkpoint-name",arm,"--checkpoint-path",target["checkpoint"]]
            with (audit/f"{arm}-prediction.log").open("a") as log:
                subprocess.run(args+["--predictions-only"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
            cpulog=files.enter_context((audit/f"{arm}-scoring.log").open("a"))
            children.append(subprocess.Popen(args+["--cpu-only"],stdout=cpulog,stderr=subprocess.STDOUT))
            print(f"Trained and extracted final predictions from {arm}.",flush=True)
        if len({j["target"]["data_sha256"] for j in models.values()})!=1:
            raise ValueError("Downstream rows differ")
        if len({j[stage]["weights_sha256"] for j in models.values() for stage in ("parent","target")})!=6:
            raise ValueError("Independent model development did not produce six distinct weight files")
        timing=dict(training_start_utc=started,final_gpu_predictions_utc=now(),
                    training_prediction_seconds=time.monotonic()-start)
        write_json(run_root/"timing.json",timing)
        returncodes=[child.wait() for child in children]
        if any(returncodes):
            raise RuntimeError(f"Confirmation CPU scoring failed: {returncodes}")
    subprocess.run([sys.executable,"-m","experiments.watermark_inheritance.matched_process_audit",
                    "--run-dir",str(run_root),"--output-dir",str(audit)],check=True)
    timing.update(verified_audit_utc=now(),elapsed_training_to_verified_audit=time.monotonic()-start)
    write_json(run_root/"timing.json",timing)
    print("All predeclared confirmation outcomes verified.",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("family",choices=FAMILIES)
    ap.add_argument("epoch",type=int,choices=PREFIXES)
    ap.add_argument("--deadline",required=True,help="ISO 8601 UTC wall-clock GPU deadline")
    ap.add_argument("--backup",action="store_true")
    args=ap.parse_args()
    location=RUN.with_name("confirmation_backup") if args.backup else RUN
    run(args.family,args.epoch,datetime.fromisoformat(args.deadline).timestamp(),location)
