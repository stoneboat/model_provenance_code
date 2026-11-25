"""Exact original-batch BF16 reading on construction rows, with reusable packed greenlists."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.score import calibrate_threshold, eligible_predictions, evaluate_shard, infer_batch

ORIGINAL = Path("artifacts/m7vv_q4")
OUT = Path("artifacts/m7vv_minimal/construction")
SIDES = ("s", "s_prime")


def cache_path():
    return OUT / "greenlists"


def score_identity(rows):
    digest = hashlib.sha256()
    for row in rows:
        if row["audit_split"]=="calibration":
            digest.update(json.dumps([row[k] for k in ("pair_id", "prompt_ids", "s_ids", "s_prime_ids")]).encode())
    return digest.hexdigest()


def init_mask_worker(tok,config,cache,shape):
    global worker_wm, worker_mask
    torch.set_num_threads(1)
    worker_wm=Maryland(tok,{**config,"watermark_cache_size":4})
    worker_mask=np.memmap(cache/"membership.bits",mode="r+",dtype=np.uint8,shape=shape)


def fill_greenlist_rows(chunk):
    first,contexts=chunk
    green=np.zeros(worker_wm.vocab_size,np.uint8)
    for i,context in enumerate(contexts,first):
        green.fill(0)
        green[worker_wm.greenlist(context).numpy()]=1
        worker_mask[i]=np.packbits(green,bitorder="little")


def prepare(rows, tok, config):
    cache = cache_path()
    if (cache/"manifest.json").exists():
        return
    cache.mkdir(exist_ok=True, parents=True)
    cal = [r for r in rows if r["audit_split"]=="calibration"]
    contexts = {}
    ctx_indices, offsets, positions = [], [0], []
    wm=Maryland(tok,{**config,"watermark_cache_size":8})
    for row in cal:
        for side in SIDES:
            ids=row[side+"_ids"]
            for context, idx in eligible_predictions(row["prompt_ids"],ids,range(len(ids)),wm.ngram):
                ctx_indices.append(contexts.setdefault(context,len(contexts)))
                positions.append(idx)
            offsets.append(len(positions))
    shape = (len(contexts), (len(tok)+7)//8)
    mask=np.memmap(cache/"membership.bits",mode="w+",dtype=np.uint8,shape=shape)
    mask.flush()
    del mask
    context_list=list(contexts)
    chunks=[(i,context_list[i:i+1024]) for i in range(0,len(context_list),1024)]
    with mp.get_context("fork").Pool(5,initializer=init_mask_worker,
                                     initargs=(tok,config,cache,shape)) as workers:
        workers.map(fill_greenlist_rows,chunks,chunksize=1)
    np.savez(cache/"positions.npz",context=np.asarray(ctx_indices,dtype=np.int32),
             token=np.asarray(positions,dtype=np.int16),offsets=np.asarray(offsets,dtype=np.int32))
    write_json(cache/"manifest.json",dict(pairs=len(cal),shape=shape,eligible=len(positions),
        gamma=wm.gamma_exact,construction_sha256=score_identity(rows),config=config))


def predict(rows, tok, checkpoint):
    model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    predictions={}
    # Retain the original dimensions, row slots and attention masks. Rows outside
    # construction carry EOS fillers of the original lengths and are discarded.
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for first in range(0,len(rows),16):
        batch=rows[first:first+16]
        if not any(r["audit_split"]=="calibration" for r in batch):
            continue
        examples=[]
        for row in batch:
            for side in SIDES:
                p,c=row["prompt_ids"],row[side+"_ids"]
                if row["audit_split"]!="calibration":
                    p,c=[pad]*len(p),[pad]*len(c)
                examples.append((p,c))
        vals=infer_batch(model,examples,"cuda:0",len(tok),pad)
        for i,row in enumerate(batch):
            if row["audit_split"]=="calibration":
                predictions[row["pair_id"]]={side:vals[2*i+j] for j,side in enumerate(SIDES)}
    return predictions


def apply_membership(rows,predictions):
    cache=cache_path()
    manifest=json.loads((cache/"manifest.json").read_text())
    if manifest["construction_sha256"]!=score_identity(rows):
        raise ValueError("Construction inputs changed")
    mask=np.memmap(cache/"membership.bits",mode="r",dtype=np.uint8,shape=tuple(manifest["shape"]))
    pos=np.load(cache/"positions.npz")
    ctx,token,offsets=pos["context"],pos["token"],pos["offsets"]
    gamma=manifest["gamma"]
    examples=[]
    for i,r in enumerate(row for row in rows if row["audit_split"]=="calibration"):
        pair=r["pair_id"]
        row={"pair_id":pair}
        for j,side in enumerate(SIDES):
            begin,end=offsets[2*i+j:2*i+j+2]
            out=predictions[pair][side]
            pred=np.array(out["predictions"],dtype=np.int32)[token[begin:end]]
            green=int(((mask[ctx[begin:end],pred//8]>>(pred%8))&1).sum())
            n=int(end-begin)
            row[side]=dict(z=(green-gamma*n)/math.sqrt(n*gamma*(1-gamma)) if n else 0.,
                  green=green,eligible=n,min_k20=out["min_k20"])
        examples.append(row)
    return examples


def summarize(examples):
    # Identity of fixed construction folds uses no scores or labels.
    perm=sorted(range(len(examples)),key=lambda i:hashlib.sha256(
        ("m7vv-split-v1:"+str(examples[i]["pair_id"])).encode()).digest())
    folds=[perm[::2],perm[1::2]]
    metrics={}
    for key in ("z","min_k20"):
        s,sp=([r[side][key] for r in examples] for side in SIDES)
        fit=calibrate_threshold(s,sp); n=len(s)
        cross=[]
        for i in (0,1):
            tr,te=folds[i],folds[1-i]
            threshold=calibrate_threshold([s[k] for k in tr],[sp[k] for k in tr])["threshold"]
            gap=evaluate_shard([s[k] for k in te],[sp[k] for k in te],threshold)["signed_gap"]
            cross.append(dict(fit_fold=i,threshold=threshold,gap=gap))
        metrics[key]=dict(fit=fit,apparent_gap=evaluate_shard(s,sp,fit["threshold"])["signed_gap"],
            auc=float((rankdata(s+sp)[:n].sum()-n*(n+1)/2)/n**2),
            mean_s=np.mean(s),mean_s_prime=np.mean(sp),crossfit=cross,
            twofold_gap=np.mean([v["gap"] for v in cross]))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--label")
    parser.add_argument("--verify-A",action="store_true")
    parser.add_argument("--prepare",action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predictions-only",action="store_true")
    mode.add_argument("--from-predictions",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    cfg=load_config(ORIGINAL)
    rows=read_jsonl(ORIGINAL/"prepared/pairs.jsonl")
    tok=AutoTokenizer.from_pretrained(cfg["tokenizer_path"],local_files_only=True)
    if args.prepare:
        prepare(rows,tok,cfg)
        # Prove all green decisions and calibrated thresholds from the cache
        # reproduce the original A results even before live checkpoint replay.
        cal_ids={r["pair_id"] for r in rows if r["audit_split"]=="calibration"}
        old={row["pair_id"]:row for row in read_jsonl(ORIGINAL/"scores/A/predictions.jsonl") if row["pair_id"] in cal_ids}
        rebuilt=apply_membership(rows,old)
        saved={r["pair_id"]:r for r in read_jsonl(ORIGINAL/"scores/A/examples.jsonl")}
        assert all(all(r[side][k]==saved[r["pair_id"]][side][k] for side in SIDES for k in ("z","green","eligible","min_k20")) for r in rebuilt)
        moments=summarize(rebuilt)
        write_json(OUT/"archived_A.json",moments)
        print("Verified all 800 archived A construction scores",flush=True)
        return
    if not args.checkpoint or not args.label:
        parser.error("--checkpoint and --label are required")
    started=time.time()
    dest=OUT/args.label
    dest.mkdir(exist_ok=True,parents=True)
    cache_file=dest/"construction_predictions.jsonl"
    if args.from_predictions:
        pred={r["pair_id"]:{side:r[side] for side in SIDES} for r in read_jsonl(cache_file)}
    else:
        pred=predict(rows,tok,args.checkpoint)
        write_jsonl(cache_file,[{"pair_id":idx,**values} for idx,values in pred.items()])
    if args.verify_A:
        original={r["pair_id"]:r for r in read_jsonl(ORIGINAL/"scores/A/predictions.jsonl")}
        if not all(record[side]==original[pair][side] for pair,record in pred.items() for side in SIDES):
            raise ValueError("Live A predictions differ from archived A")
    if args.predictions_only:
        print(f"Completed {len(pred)} construction predictions for {args.label}",flush=True)
        return
    ex=apply_membership(rows,pred)
    result=dict(label=args.label,checkpoint=str(args.checkpoint),construction_pairs=len(ex),
                metrics=summarize(ex),wall_seconds=time.time()-started,verified_live_A=args.verify_A)
    dest=OUT /args.label
    dest.mkdir(exist_ok=True,parents=True)
    write_jsonl(dest/"scores.jsonl",ex)
    write_json(dest/"summary.json",result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()
