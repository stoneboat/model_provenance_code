"""Prepare the fixed WikiText prompts and generate three Maryland draws per prompt for Q4."""
import argparse
from collections import Counter
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import Maryland, load_config, read_jsonl, write_json, write_jsonl
from .generate import configured_shard_seed, pair_record, sample_batch, sampling_seed, validate_pair, watermark_config_id
from .null_control import double_prime_seed, validate_double_prime
from .prepare import DATASETS, file_sha256, select_prompts, validate_tokenizer, verify_upstream


def prepare(run):
    archive = Path(__file__).parent / "results/2026-10-06/run_00/config.json"
    import json
    cfg = json.loads(archive.read_text())
    cfg.update(model_id="EleutherAI/pythia-160m", model_revision="50f5173d932e8e61f858120bcb800b97af589f46",
               model_path=str(run / "base"), tokenizer_path=str(run / "base"),
               upstream_path=str(run / "sources/radioactive-watermark"), shared_dir=str(run / "sources"),
               pairs=2000, calibration_pairs=400, pilot_pairs=16, max_new_tokens=256, pilot_max_new_tokens=64, generation_batch_size=48,
               radio_token_budget=4000, score_batch_size=32, run_id="m7vv_q4_160m", primary_checkpoint_name="A")
    for key in ("deadline_utc","code_commit"):
        cfg.pop(key,None)
    write_json(run / "config.json",cfg)
    verify_upstream(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"],local_files_only=True)
    validate_tokenizer(tok)
    inputs = sorted((run / "sources").glob("train-*-of-00002.parquet"))
    expected = ["74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212","ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"]
    assert [file_sha256(p) for p in inputs]==expected
    rows = (x for p in inputs for b in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=["text"]) for x in b.to_pylist())
    main,pilot,meta = select_prompts(rows,tok,count=2000,pilot_count=16,calibration_count=400)
    assert len({p["article_id"] for p in main+pilot})==2016
    for name,ps in (("prompts",main),("pilot_prompts",pilot)):
        write_jsonl(run / "prepared" / (name+".jsonl"),ps)
    write_json(run / "prepared/prompts.manifest.json",dict(**meta,source=DATASETS["wikitext"],parquet_sha256=expected,
                                                            parity="Fresh triple generation. No prior watermark pairs in local caches."))


def generate(run,phase):
    cfg = load_config(run)
    if phase == "pilot":
        cfg["max_new_tokens"] = cfg["pilot_max_new_tokens"]
    prefix = "pilot_" if phase == "pilot" else ""
    out = run / "prepared"
    prompts = read_jsonl(out / (prefix+"prompts.jsonl"))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"],local_files_only=True)
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"],torch_dtype=torch.bfloat16,local_files_only=True).to("cuda:0").eval()
    wm = Maryland(tok,cfg)
    seed = configured_shard_seed(cfg)
    pairs, third = [],[]
    start=time.monotonic()
    for i in range(0,len(prompts),cfg["generation_batch_size"]//3):
        batch=prompts[i:i+cfg["generation_batch_size"]//3]
        streams=[sampling_seed(seed,phase,p["pair_id"],side) for p in batch for side in (0,1,"s_double_prime")]
        ids=[p["prompt_ids"] for p in batch for _ in range(3)]
        outputs=sample_batch(model,tok,wm,ids,streams,cfg,"cuda:0")
        for j,p in enumerate(batch):
            a,b,c=outputs[3*j:3*j+3]
            row=pair_record(p,[a,b],tok,cfg,phase)
            validate_pair(row,p,cfg,phase,len(tok))
            r=dict(pair_id=p["pair_id"],source_id=p["source_id"],audit_split=p["audit_split"],
                   completion_ids=c,completion_length=len(c),sampling_seed=double_prime_seed(cfg,p["pair_id"],phase),
                   watermark_config_id=watermark_config_id(cfg),completion_text=tok.decode(c,skip_special_tokens=True))
            if phase=="full":
                validate_double_prime(r,row,cfg)
            pairs.append(row)
            third.append(r)
        elapsed=time.monotonic()-start
        print(f"{phase}: {len(pairs)}/{len(prompts)} triples, {elapsed:.1f}s",flush=True)
    assert len({r["sampling_seed"] for r in third}|{p[key] for p in pairs for key in ("sampling_seed_s","sampling_seed_s_prime")})==3*len(pairs)
    for name,rs in (("pairs",pairs),("s_double_prime",third)):
        write_jsonl(out/(prefix+name+".jsonl"),rs)
    for name,ids in (("s",[p["s_ids"] for p in pairs]),("s_double_prime",[r["completion_ids"] for r in third])):
        write_jsonl(out/(prefix+name+"_train.jsonl"),[dict(pair_id=p["pair_id"],prompt_ids=p["prompt_ids"],s_ids=x) for p,x in zip(prompts,ids)])
    write_json(out/(prefix+"shards.manifest.json"),dict(phase=phase,split_counts=Counter(p["audit_split"] for p in pairs),pairs=len(pairs),
        elapsed_seconds=elapsed,cap=cfg["max_new_tokens"],prompt_source="WikiText-103-raw-v1 train",model_dtype="bfloat16",
        total_tokens_by_side=dict(s=sum(p["s_length"] for p in pairs),s_prime=sum(p["s_prime_length"] for p in pairs),s_double_prime=sum(r["completion_length"] for r in third)),
        duplicate_pairs=sum(p["tied"] for p in pairs),sampling_streams=3*len(pairs),seeds_disjoint=True,
        prompt_file_sha256=file_sha256(out/(prefix+"prompts.jsonl")),
        files={n:file_sha256(out/(prefix+n+".jsonl")) for n in ("pairs","s_double_prime","s_train","s_double_prime_train")}))


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("phase",choices=["prepare","pilot","full"])
    arg.add_argument("--run",type=Path,default=Path("artifacts/m7vv_q4"))
    opts=arg.parse_args()
    if opts.phase=="prepare":
        prepare(opts.run)
    else:
        generate(opts.run,opts.phase)
