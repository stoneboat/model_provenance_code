"""Stream an exact CPU greenlist index, then replay trained-target predictions without GPU inference."""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.score import calibrate_threshold, eligible_predictions, evaluate_shard, example_watermark_score
from experiments.watermark_inheritance.calibration_analysis import paired_cutoff

SIDES=("s","s_prime")
FIRST=Path("artifacts/m7vv_minimal/confirmation")


def init_worker(tok,config,path):
    global watermarker, store, stride
    torch.set_num_threads(1)
    watermarker=Maryland(tok,{**config,"watermark_cache_size":4})
    store=path
    stride=(len(tok)+7)//8


def write_block(args):
    first,contexts=args
    mask=np.zeros(watermarker.vocab_size,dtype=np.uint8)
    with store.open("r+b") as stream:
        stream.seek(first*stride)
        for context in contexts:
            mask.fill(0)
            mask[watermarker.greenlist(context).numpy()]=1
            stream.write(np.packbits(mask,bitorder="little").tobytes())


def build(run):
    cfg=load_config(run)
    tok=AutoTokenizer.from_pretrained(cfg["tokenizer_path"],local_files_only=True)
    actual=(cfg["watermark_key"],cfg["salt_key"],cfg["gamma"],cfg["ngram"])
    assert actual==(101,35317,.25,2) and len(tok)==50277
    out=run/"greenlist_cache"
    out.mkdir(exist_ok=True,parents=True)
    bits=out/"membership.bits"
    bits.touch(exist_ok=True)
    contexts={}
    stride=(len(tok)+7)//8
    start=time.monotonic()
    bsize=16
    with mp.get_context("fork").Pool(4,initializer=init_worker,initargs=(tok,cfg,bits)) as pool:
        for first in range(0,cfg["pairs"],bsize):
            batch=run/"batches"/f"batch_{first:04d}.jsonl"
            dest=out/f"index_{first:04d}.npz"
            if dest.exists():
                with np.load(dest) as old:
                    for key in old["new_keys"]:
                        contexts[tuple(key.tolist())]=len(contexts)
                continue
            while not batch.exists():
                time.sleep(3)
            pairs=[r["pair"] for r in read_jsonl(batch)]
            assert len(pairs)==min(bsize,cfg["pairs"]-first)
            added=[]
            where=[]; positions=[]; offsets=[0]
            for row in pairs:
                for side in SIDES:
                    completion=row[side+"_ids"]
                    for key,pos in eligible_predictions(row["prompt_ids"],completion,range(len(completion)),cfg["ngram"]):
                        if key not in contexts:
                            contexts[key]=len(contexts)
                            added.append(key)
                        where.append(contexts[key]);positions.append(pos)
                    offsets.append(len(where))
            with bits.open("r+b") as stream:
                stream.truncate(len(contexts)*stride)
            begin=len(contexts)-len(added)
            pool.map(write_block,[(begin+i,added[i:i+256]) for i in range(0,len(added),256)],chunksize=1)
            tmp=dest.with_suffix(".writing")
            with tmp.open("wb") as stream:
                np.savez(stream,new_keys=np.array(added,dtype=np.int32),idx=np.array(where,dtype=np.int32),
                         pos=np.array(positions,dtype=np.int16),offset=np.array(offsets,dtype=np.int32),
                         labels=np.array([p["audit_split"] for p in pairs]),pair=np.array([p["pair_id"] for p in pairs]))
            tmp.replace(dest)
    write_json(out/"ready.json",dict(config={k:cfg[k] for k in ("watermark_key","salt_key","gamma","ngram")},
                contexts=len(contexts),stride=stride,n=cfg["pairs"],vocab=len(tok),
                gamma_exact=int(cfg["gamma"]*len(tok))/len(tok),wall_seconds=time.monotonic()-start))


def verify(run,target):
    cache=run/"greenlist_cache"
    manifest=json.loads((cache/"ready.json").read_text())
    records=read_jsonl(run/"scores"/target/"predictions.jsonl")
    pairs=read_jsonl(run/"prepared/pairs.jsonl")
    assert len(records)==len(pairs)==manifest["n"]
    membership=np.memmap(cache/"membership.bits",mode="r",dtype=np.uint8,
                         shape=(manifest["contexts"],manifest["stride"]))
    gamma=manifest["gamma_exact"]
    scores=[]
    for first in range(0,len(records),16):
        with np.load(cache/f"index_{first:04d}.npz") as data:
            offsets,where,positions=data["offset"],data["idx"],data["pos"]
            for j,(pair_id,split) in enumerate(zip(data["pair"],data["labels"])):
                pred=records[first+j];pair=pairs[first+j]
                assert pair["pair_id"]==pred["pair_id"]==pair_id and split==pair["audit_split"]
                row=dict(pair_id=pair_id,audit_split=split)
                for i,side in enumerate(SIDES):
                    lo,hi=offsets[2*j+i:2*j+i+2]; n=int(hi-lo)
                    ids=np.array(pred[side]["predictions"],dtype=np.int32)
                    assert len(ids)==len(pair[side+"_ids"])
                    picked=ids[positions[lo:hi]]
                    green=int(((membership[where[lo:hi],picked//8]>>(picked%8))&1).sum())
                    z=(green-n*gamma)/math.sqrt(n*gamma*(1-gamma)) if n else 0.
                    row[side]=dict(z=z,green=green,eligible=n,min_k20=pred[side]["min_k20"])
                scores.append(row)
    # Spot-check exact original per-completion scoring on first and last pair
    # of each split prior to releasing the fast decisions.
    tok=AutoTokenizer.from_pretrained(load_config(run)["tokenizer_path"],local_files_only=True)
    wm=Maryland(tok,load_config(run))
    checked=[]
    for split in ("calibration","audit"):
        indices=[i for i,p in enumerate(pairs) if p["audit_split"]==split]
        for idx in (indices[0],indices[-1]):
            checked.append(pairs[idx]["pair_id"])
            for side in SIDES:
                value,_=example_watermark_score(pairs[idx]["prompt_ids"],pairs[idx][side+"_ids"],
                                                records[idx][side]["predictions"],wm)
                assert all(value[key]==scores[idx][side][key] for key in ("eligible","green","z"))
    out=run/"fast_replay"/target
    out.mkdir(exist_ok=True,parents=True)
    write_json(out/"original_spotcheck.json",dict(pair_ids=checked,completions=2*len(checked),exact=True))
    write_jsonl(out/"scores.jsonl",scores)
    summary={}
    for kind,field in [("watermark","z"),("min_k20","min_k20")]:
        def side_scores(split):
            return [[row[side][field] for row in scores if row["audit_split"]==split] for side in SIDES]
        fit=calibrate_threshold(*side_scores("calibration")); assert fit["calibration_pairs"]==400
        a,b=side_scores("audit");t=fit["threshold"];res=evaluate_shard(a,b,t)
        assert res["audit_pairs"]==1600
        count_a=sum(v>t for v in a);count_b=sum(v>t for v in b)
        gap=(count_a-count_b)/1600;crit=paired_cutoff(.05,1600)
        summary[kind]=dict(fit=fit,evaluation=res,accepted_S=count_a,accepted_S_prime=count_b,
                           precise_signed_gap=gap,cutoff=crit,rejects=abs(gap)>crit)
    write_json(out/"summary.json",summary)


def compare_original(run):
    verified={}
    for target in ("A","B","C"):
        fast=run/"fast_replay"/target
        original=run/"scores"/target
        ours=read_jsonl(fast/"scores.jsonl")
        theirs=read_jsonl(original/"examples.jsonl")
        assert len(ours)==len(theirs)==2000
        for a,b in zip(ours,theirs,strict=True):
            assert a["pair_id"]==b["pair_id"] and a["audit_split"]==b["audit_split"]
            assert all(a[side][k]==b[side][k] for side in SIDES for k in ("z","min_k20","eligible","green"))
        stats=json.loads((fast/"summary.json").read_text())
        refs=json.loads((original/"metrics.json").read_text())
        for key,summary in stats.items():
            assert dict(threshold=summary["fit"]["threshold"],**summary["evaluation"])==refs["shard"][key]
        verified[target]=dict(examples=4000,detectors=2,paired_decisions=1600*2,exact=True)
    write_json(run/"fast_replay"/"original_parity.json",verified)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","replay","compare-original"))
    p.add_argument("--run",type=Path,default=FIRST)
    p.add_argument("--target",choices=("A","B","C","P","Q","R"))
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.action=="prepare":
        build(args.run)
    elif args.action=="replay":
        assert args.target
        verify(args.run,args.target)
    else:
        compare_original(args.run)

if __name__=="__main__":
    main()
