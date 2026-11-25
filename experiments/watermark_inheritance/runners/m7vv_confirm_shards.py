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
    configured_shard_seed, double_prime_seed, pair_record, sample_batch, sampling_seed,
    validate_double_prime, validate_pair, watermark_config_id,
)
from experiments.watermark_inheritance.prepare import encode_dolly, file_sha256, select_prompts, validate_tokenizer, verify_upstream

RUN = Path("artifacts/m7vv_witness_sharing")
PROTOCOL = Path(__file__).resolve().parents[1] / "configs/m7vv_witness_sharing.json"


def prepare(run, inputs, dolly_file):
    """Recover the published prompt order and fixed downstream training rows."""
    protocol = json.loads(PROTOCOL.read_text())
    cfg = dict(protocol["generation"], model_path=str(inputs / "base"),
               tokenizer_path=str(inputs / "base"), shared_dir=str(inputs / "sources"),
               upstream_path=str(inputs / "sources/radioactive-watermark"))
    verify_upstream(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True)
    validate_tokenizer(tokenizer)
    parquet = sorted((inputs / "sources").glob("train-*-of-00002.parquet"))
    wikitext = protocol["wikitext"]
    if [file_sha256(path) for path in parquet] != wikitext["parquet_sha256"]:
        raise ValueError("Expected both pinned WikiText parquet files")

    def select(count, pilot_count, seed):
        rows = (row for path in parquet for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8192, columns=["text"]) for row in batch.to_pylist())
        main, pilot, _ = select_prompts(rows, tokenizer, count=count, pilot_count=pilot_count,
                                       calibration_count=0, prompt_tokens=cfg["prompt_tokens"], seed=seed)
        return main + pilot

    def exclude_previous(candidates, previous):
        article_ids = {row["article_id"] for row in previous}
        prefixes = {tuple(row["prompt_ids"]) for row in previous}
        return [row for row in candidates if row["article_id"] not in article_ids
                and tuple(row["prompt_ids"]) not in prefixes]

    # Reconstruct selections only. Earlier completions and checkpoints are unnecessary.
    first, second = wikitext["excluded_prompt_draws"]
    count = cfg["pairs"]
    previous = select(count, first["pilot_count"], first["prompt_seed"])
    previous += exclude_previous(select(count + len(previous), 0, second["prompt_seed"]), previous)[:count]
    prompts = exclude_previous(select(count + len(previous), 0, cfg["prompt_seed"]), previous)[:count]
    construction = set(random.Random(wikitext["construction_assignment_seed"]).sample(
        range(count), cfg["calibration_pairs"]))
    for index, row in enumerate(prompts):
        row.update(pair_id=f"backup_{index:05d}", audit_split="calibration" if index in construction else "audit")
    if len(prompts) != count:
        raise ValueError("Insufficient disjoint WikiText prompts")

    prepared = run / "prepared"
    prompts_file = prepared / "prompts.jsonl"
    write_jsonl(prompts_file, prompts)
    prompts_digest = file_sha256(prompts_file)
    if prompts_digest != wikitext["published_prompts_sha256"]:
        raise ValueError("Prompt order or tokens differ from the published run")
    write_json(prepared / "prompts.manifest.json", {
        "prompts_sha256": prompts_digest,
        "excluded_article_count": len(previous),
        "split_counts": Counter(row["audit_split"] for row in prompts),
    })

    dolly = protocol["dolly"]
    if file_sha256(dolly_file) != dolly["source_sha256"]:
        raise ValueError("Dolly file differs from the pinned source")
    rows = read_jsonl(dolly_file)
    if len(rows) != dolly["source_rows"]:
        raise ValueError("Unexpected Dolly row count")
    indices = random.Random(dolly["selection_seed"]).sample(range(len(rows)), dolly["rows"])
    encoded = []
    for index in indices:
        example = encode_dolly(rows[index], tokenizer, max_length=protocol["training"]["max_length"])
        if example is None:
            raise ValueError(f"Invalid Dolly row {index}")
        encoded.append(dict(source_index=index, **example))
    dolly_path = prepared / "dolly.jsonl"
    write_jsonl(dolly_path, encoded)
    output_digest = file_sha256(dolly_path)
    if output_digest != dolly["encoded_sha256"]:
        raise ValueError("Dolly preprocessing differs from the published run")
    write_json(prepared / "dolly_manifest.json", {
        "selection": "fixed_source_rows",
        "selection_seed": dolly["selection_seed"],
        "dataset_id": dolly["id"],
        "dataset_revision": dolly["revision"],
        "rows": len(encoded),
        "output_file_sha256": output_digest,
    })
    write_json(run / "config.json", cfg)
    write_json(run / "protocol.json", protocol)
    print(f"Verified {len(prompts)} published prompts and {len(encoded)} Dolly rows.", flush=True)


def generate(max_batches, run=RUN):
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
    first, second = json.loads(PROTOCOL.read_text())["wikitext"]["excluded_prompt_draws"]
    prior_draws = (
        (first["shard_seed"], "full", "pair", len(prompts)),
        (first["shard_seed"], "pilot", "pilot", first["pilot_count"]),
        (second["shard_seed"], "full", "confirm", len(prompts)),
    )
    previous_streams = {
        sampling_seed(seed, phase, f"{prefix}_{index:05d}", side)
        for seed, phase, prefix, count in prior_draws for index in range(count)
        for side in (0, 1, "s_double_prime")
    }
    if len(streams) != 3 * len(prompts) or streams & previous_streams:
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
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--dolly-file", type=Path)
    args = parser.parse_args()
    inputs = (args.inputs or args.run_dir / "inputs").resolve()
    if args.action == "prepare":
        prepare(args.run_dir, inputs, args.dolly_file or inputs / "sources/databricks-dolly-15k.jsonl")
    else:
        generate(args.max_batches, args.run_dir)
