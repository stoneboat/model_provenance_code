"""Generate disjoint, independently seeded confirmation triples using the original sampler."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.generate import (
    configured_shard_seed, pair_record, sample_batch, sampling_seed, validate_pair, watermark_config_id,
)
from experiments.watermark_inheritance.null_control import double_prime_seed, validate_double_prime
from experiments.watermark_inheritance.prepare import file_sha256, select_prompts, validate_tokenizer, verify_upstream

ORIGINAL = Path("artifacts/m7vv_q4")
RUN = Path("artifacts/m7vv_minimal/confirmation")
PROMPT_SEED = 21729
SPLIT_SEED = 27183
SHARD_SEED = 20301


def prepare(run=RUN, backup=False):
    prepared = run / "prepared"
    prompt_seed, split_seed, shard_seed = ((31729, 37183, 30301) if backup else (PROMPT_SEED, SPLIT_SEED, SHARD_SEED))
    if (prepared / "prompts.manifest.json").exists():
        raise FileExistsError("Confirmation prompts have already been prepared")
    cfg = load_config(ORIGINAL)
    cfg.update(run_id="m7vv_independent_backup" if backup else "m7vv_independent_confirmation", seed=shard_seed,
               shard_seed=shard_seed, prompt_seed=prompt_seed, pilot_pairs=0)
    write_json(run / "config.json", cfg)
    verify_upstream(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True)
    validate_tokenizer(tok)
    old_rows = read_jsonl(ORIGINAL / "prepared/prompts.jsonl") + read_jsonl(ORIGINAL / "prepared/pilot_prompts.jsonl")
    if backup:
        old_rows += read_jsonl(RUN / "prepared/prompts.jsonl")
    old_articles = {r["article_id"] for r in old_rows}
    old_prefixes = {tuple(r["prompt_ids"]) for r in old_rows}
    columns = ["text"]
    parquet = sorted((ORIGINAL / "sources").glob("train-*-of-00002.parquet"))
    manifest = json.loads((ORIGINAL / "prepared/prompts.manifest.json").read_text())
    if [file_sha256(path) for path in parquet] != manifest["parquet_sha256"]:
        raise ValueError("WikiText input revisions differ from original run")
    data = (r for path in parquet for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=columns)
            for r in batch.to_pylist())
    n = cfg["pairs"]
    pool, _, info = select_prompts(data, tok, count=n+len(old_rows), pilot_count=0,
                                    calibration_count=0, prompt_tokens=cfg["prompt_tokens"], seed=prompt_seed)
    main = [p for p in pool if p["article_id"] not in old_articles and tuple(p["prompt_ids"]) not in old_prefixes][:n]
    calibration = set(random.Random(split_seed).sample(range(n),cfg["calibration_pairs"]))
    if len(main) != n or len({p["article_id"] for p in main}) != n:
        raise ValueError("Independent prompt coverage mismatch")
    for i, p in enumerate(main):
        p.update(pair_id=f"backup_{i:05d}" if backup else f"confirm_{i:05d}", audit_split="calibration" if i in calibration else "audit")
    path = prepared / "prompts.jsonl"
    write_jsonl(path,main)
    write_json(prepared / "prompts.manifest.json", dict(
        source=manifest["source"], parquet_sha256=manifest["parquet_sha256"], source_pool=info["eligible_articles"],
        split_seed=split_seed,prompt_seed=prompt_seed,shard_seed=shard_seed,reused_watermark_key=cfg["watermark_key"],
        distinct_articles=n,overlapping_old_articles=0,overlapping_old_token_prefixes=0,
        excluded_original_article_count=len(old_articles),split_counts=Counter(p["audit_split"] for p in main),
        prompts_sha256=file_sha256(path), max_new_tokens=cfg["max_new_tokens"]))
    print(f"Prepared {n} distinct fresh prompts and split; {len(old_articles)} development/pilot articles excluded.",flush=True)


def generate(max_batches, run=RUN, backup=False):
    prepared = run / "prepared"
    cfg = load_config(run)
    info=json.loads((prepared/"prompts.manifest.json").read_text())
    if info["prompts_sha256"]!=file_sha256(prepared/"prompts.jsonl"):
        raise ValueError("Confirmation prompts changed")
    prompts = read_jsonl(prepared / "prompts.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"],local_files_only=True)
    torch.set_num_threads(1)
    model = None
    wm = None
    batches = run / "batches"
    new_batches=0
    seed=configured_shard_seed(cfg)
    width=cfg["generation_batch_size"]//3
    pairs=[]
    thirds=[]
    started=time.monotonic()
    for i in range(0,len(prompts),width):
        destination=batches/f"batch_{i:04d}.jsonl"
        batch=prompts[i:i+width]
        if destination.exists():
            triple=read_jsonl(destination)
        else:
            if (run/"pause.request").exists() or (max_batches and new_batches>=max_batches):
                print(f"Paused after {i} verified triples.",flush=True)
                return
            if model is None:
                model=AutoModelForCausalLM.from_pretrained(cfg["model_path"],torch_dtype=torch.bfloat16,local_files_only=True).to("cuda:0").eval()
                wm=Maryland(tok,cfg)
            streams=[sampling_seed(seed,"full",p["pair_id"],side) for p in batch for side in (0,1,"s_double_prime")]
            inputs=[p["prompt_ids"] for p in batch for _ in range(3)]
            drawn=sample_batch(model,tok,wm,inputs,streams,cfg,"cuda:0")
            triple=[]
            for j,p in enumerate(batch):
                a,b,c=drawn[3*j:3*j+3]
                pair=pair_record(p,[a,b],tok,cfg,"full")
                third=dict(pair_id=p["pair_id"],source_id=p["source_id"],audit_split=p["audit_split"],
                    completion_ids=c,completion_length=len(c),sampling_seed=double_prime_seed(cfg,p["pair_id"],"full"),
                    watermark_config_id=watermark_config_id(cfg),completion_text=tok.decode(c,skip_special_tokens=True))
                triple.append({"pair":pair,"third":third})
            write_jsonl(destination,triple)
            new_batches+=1
            print(f"Generated {i+len(batch)}/{len(prompts)} triples; {time.monotonic()-started:.1f}s in session.",flush=True)
        if len(triple)!=len(batch):
            raise ValueError("Incomplete chunk")
        for p,r in zip(batch,triple):
            validate_pair(r["pair"],p,cfg,"full",len(tok))
            validate_double_prime(r["third"],r["pair"],cfg)
            pairs.append(r["pair"])
            thirds.append(r["third"])
    streams={r["sampling_seed"] for r in thirds} | {p[k] for p in pairs for k in ("sampling_seed_s","sampling_seed_s_prime")}
    archived_streams = set()
    for prefix in ("","pilot_"):
        old=ORIGINAL/"prepared"
        archived_streams.update(r["sampling_seed"] for r in read_jsonl(old/(prefix+"s_double_prime.jsonl")))
        archived_streams.update(p[k] for p in read_jsonl(old/(prefix+"pairs.jsonl")) for k in ("sampling_seed_s","sampling_seed_s_prime"))
    if backup:
        first=RUN/"prepared"
        archived_streams.update(r["sampling_seed"] for r in read_jsonl(first/"s_double_prime.jsonl"))
        archived_streams.update(p[k] for p in read_jsonl(first/"pairs.jsonl") for k in ("sampling_seed_s","sampling_seed_s_prime"))
    if len(streams)!=3*len(prompts) or (streams & archived_streams):
        raise ValueError("Sampling streams overlap")
    write_jsonl(prepared/"pairs.jsonl",pairs)
    write_jsonl(prepared/"s_double_prime.jsonl",thirds)
    write_jsonl(prepared/"s_train.jsonl",[dict(pair_id=p["pair_id"],prompt_ids=p["prompt_ids"],s_ids=p["s_ids"]) for p in pairs])
    write_jsonl(prepared/"s_double_prime_train.jsonl",[dict(pair_id=p["pair_id"],prompt_ids=p["prompt_ids"],s_ids=r["completion_ids"]) for p,r in zip(pairs,thirds)])
    write_json(prepared/"shards.manifest.json",dict(pairs=len(pairs),total_draws=len(streams),shard_seed=seed,
        split_counts=Counter(p["audit_split"] for p in pairs),overlap_with_original_rng_streams=0,
        s_sprime_ties=sum(p["tied"] for p in pairs),s_sdoubleprime_ties=sum(p["s_ids"]==r["completion_ids"] for p,r in zip(pairs,thirds)),
        sprime_sdoubleprime_ties=sum(p["s_prime_ids"]==r["completion_ids"] for p,r in zip(pairs,thirds)),
        files={f:file_sha256(prepared/(f+".jsonl")) for f in ("pairs","s_double_prime","s_train","s_double_prime_train")}))
    print("Verified and finalized confirmation shards.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","generate"])
    parser.add_argument("--max-batches",type=int)
    parser.add_argument("--backup",action="store_true",help="Use disjoint, prespecified backup inputs and seeds")
    args=parser.parse_args()
    location=RUN.with_name("confirmation_backup") if args.backup else RUN
    if args.action=="prepare":
        prepare(location,args.backup)
    else:
        generate(args.max_batches,location,args.backup)
