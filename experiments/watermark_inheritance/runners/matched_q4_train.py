"""Prepare a shard-independent Dolly subset or train one matched full-parameter target."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import time

from transformers import AutoTokenizer

from experiments.watermark_inheritance.common import load_json, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.prepare import encode_dolly, file_sha256
from experiments.watermark_inheritance.train import train_stage

DOLLY = Path("artifacts/dolly_matched/data/databricks-dolly-15k.jsonl")
BASE_REV = "50f5173d932e8e61f858120bcb800b97af589f46"
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
MODEL_SEEDS = {"A": 34711, "B": 45931, "C": 58271}
DOLLY_SEED = 31793


def prepare(root):
    base = root / "base"
    rows = read_jsonl(DOLLY)
    assert len(rows) == 15011 and load_json(DOLLY.with_name("source.json"))["revision"] == DOLLY_REV
    indices = random.Random(DOLLY_SEED).sample(range(len(rows)), 512)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    records = []
    for i in indices:
        ex = encode_dolly(rows[i], tokenizer, max_length=512)
        if ex is None:
            raise ValueError(f"Invalid fixed Dolly source row {i}")
        records.append(dict(source_index=i, **ex))
    shared = root / "train/shared"
    data = shared / "dolly.jsonl"
    write_jsonl(data, records)
    write_json(shared / "dolly_manifest.json", {
        "selection":"fixed_source_rows", "selection_seed":DOLLY_SEED,"selection_function":"random.Random(seed).sample(range(15011),512)",
        "dataset_id":"databricks/databricks-dolly-15k","dataset_revision":DOLLY_REV,"source_file":str(DOLLY),
        "source_file_sha256":file_sha256(DOLLY), "rows":len(records),"source_indices":indices,
        "formatter":"experiments.watermark_inheritance.prepare.encode_dolly","max_length":512,
        "output_file_sha256":file_sha256(data),"response_tokens":sum(r["response_tokens"] for r in records),
        "truncated_responses":sum(r["response_truncated"] for r in records)})
    for name,seed in MODEL_SEEDS.items():
        run = root / "train" / name
        prep = run / "prepared"
        prep.mkdir(exist_ok=True,parents=True)
        shard = "s_double_prime_train.jsonl" if name=="C" else "s_train.jsonl"
        for dst, src in [("pairs.jsonl",root/"prepared"/shard),("dolly.jsonl",data),("dolly_manifest.json",shared/"dolly_manifest.json")]:
            link = prep / dst
            if link.is_symlink():
                assert link.resolve()==src.resolve()
            else:
                link.symlink_to(src.resolve())
        config = dict(experiment="matched_q4",arm=name,model_id="EleutherAI/pythia-160m", model_revision=BASE_REV,
            model_path=str(base.resolve()), tokenizer_path=str(base.resolve()),pairs=2000,seed=seed,model_seed=seed,
            stage_epochs={"parent":3,"target":1},effective_batch_size=32,train_micro_batch=8,max_length=512,
            learning_rate=1e-5,weight_decay=0.01,warmup_ratio=0.03,max_grad_norm=1.0,
            dolly_selection="fixed_source_rows",dolly_selection_seed=DOLLY_SEED,dolly_revision=DOLLY_REV,
            save_final_only=True, cpu_threads=3, log_every=10)
        path = run / "config.json"
        if path.exists() and load_json(path)!=config:
            raise ValueError("Existing arm has a different configuration")
        write_json(path, config)


def run_arm(root,arm):
    run = root / "train" / arm
    config = load_json(run / "config.json")
    rows = read_jsonl(run / "prepared/pairs.jsonl")
    assert len(rows)==2000 and len({r["pair_id"] for r in rows})==2000
    if any(set(r)-{"pair_id","prompt_ids","s_ids"} for r in rows):
        raise ValueError("Training accepts only the arm-specific witness projection")
    ledger_path=run/"lineage.json"
    ledger=load_json(ledger_path) if ledger_path.exists() else {}
    parent = run / "checkpoints/parent"
    for stage,source,dest,offset in [
        ("parent",root/"base",parent,101),
        ("target",parent,run/"checkpoints/target_epoch1",202),
    ]:
        if stage in ledger and (dest/"stage_state.json").exists():
            assert file_sha256(dest/"model.safetensors")==ledger[stage]["output_model_sha256"]
            continue
        started=datetime.now(timezone.utc).isoformat()
        begin=time.monotonic()
        before=file_sha256(source/"model.safetensors")
        stats=train_stage(run,stage,"cuda:0")
        assert stats["complete"] and stats["global_step"]==(189 if stage=="parent" else 16)
        ledger[stage] = dict(start_utc=started,end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-begin,
            arm=arm,stage=stage,seed=config["model_seed"]+offset,order_seed_per_epoch="stage seed + 1-based epoch",
            updates=stats["global_step"], source_model=str(source),source_model_sha256=before,
            output_model=str(dest),output_model_sha256=file_sha256(dest/"model.safetensors"),
            training_data_sha256=stats["identity"]["data_sha256"], epochs=stats["history"])
        write_json(ledger_path,ledger)
    assert ledger["target"]["source_model_sha256"]==ledger["parent"]["output_model_sha256"]
    print(f"Completed matched arm {arm}",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/m7vv_q4"))
    parser.add_argument("action",choices=["prepare",*MODEL_SEEDS])
    args=parser.parse_args()
    if args.action=="prepare":
        prepare(args.root)
    else:
        run_arm(args.root,args.action)
