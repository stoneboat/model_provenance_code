"""Inference-only raw reading scores on the fresh S-double-prime shard."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.score import example_watermark_score, file_digest, infer_batch

TARGETS = ("A", "B", "C")


def load_inputs(run):
    prompts = read_jsonl(run / "prepared/prompts.jsonl")
    fresh = read_jsonl(run / "prepared/s_double_prime.jsonl")
    if len(fresh) != 2000 or len(prompts) != len(fresh):
        raise ValueError("Incomplete prompt or fresh completion stream")
    for p, f in zip(prompts, fresh):
        if any(p[k] != f[k] for k in ("pair_id", "source_id", "audit_split")):
            raise ValueError("Fresh shard does not match prompt ordering")
    return prompts, fresh


def predict(run, out):
    cfg = load_config(run)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompts, fresh = load_inputs(run)
    length = cfg["score_batch_size"]
    reference = read_jsonl(run / "prepared/pairs.jsonl")[:length//2]
    parity_examples = [(p["prompt_ids"], p[side+"_ids"]) for p in reference for side in ("s","s_prime")]
    parity = {}
    for target in TARGETS:
        ckpt = run / "train" / target / "checkpoints/target_epoch1"
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                       local_files_only=True).to("cuda:0").eval()
        model.requires_grad_(False)
        cached = read_jsonl(run / "scores" / target / "predictions.jsonl")[:len(reference)]
        assert all(p["pair_id"] == r["pair_id"] for p, r in zip(reference, cached))
        actual = infer_batch(model, parity_examples, "cuda:0", len(tok), pad)
        expected = [r[side] for r in cached for side in ("s", "s_prime")]
        assert actual == expected, f"Original inference batch differs for {target}"
        parity[target] = {"exact_cached_completion_replays": len(expected),
                          "checkpoint_sha256": file_digest(ckpt/"model.safetensors")}
        output = []
        for i in range(0,len(prompts),length):
            examples = [(p["prompt_ids"], f["completion_ids"])
                        for p,f in zip(prompts[i:i+length],fresh[i:i+length])]
            batch = infer_batch(model, examples, "cuda:0", len(tok), pad)
            output.extend({"pair_id": p["pair_id"], **pred} for p,pred in zip(prompts[i:i+length],batch))
        write_jsonl(out / f"{target}_predictions.jsonl", output)
        print(f"{target}: reproduced original batch and predicted {len(output)} fresh completions", flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(out/"inference.json", dict(device="cuda:0",dtype="bfloat16",backend="sdpa",batch_size=length,
               pairs=2000,parity=parity,fresh_input_sha256=file_digest(run/"prepared/s_double_prime.jsonl")))


def score_chunk(args):
    cfg, tasks = args
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True)
    wm = Maryland(tok,cfg)
    output = []
    for p, f, predictions in tasks:
        ids = f["completion_ids"]
        item = dict(pair_id=p["pair_id"], audit_split=p["audit_split"], source_id=p["source_id"])
        observed, _ = example_watermark_score(p["prompt_ids"],ids,ids,wm)
        item["observed"] = observed
        for target, pred in zip(TARGETS,predictions):
            if pred["pair_id"] != p["pair_id"] or len(ids) != pred["completion_tokens"]:
                raise ValueError("Fresh-shard prediction identity or length mismatch")
            raw, _ = example_watermark_score(p["prompt_ids"],ids,pred["predictions"],wm)
            item[target] = dict(**raw, min_k20=pred["min_k20"],completion_tokens=len(ids))
        output.append(item)
    return output


def score(run, out, workers):
    cfg = load_config(run)
    prompts, fresh = load_inputs(run)
    predictions = [read_jsonl(out / f"{target}_predictions.jsonl") for target in TARGETS]
    if any(len(preds) != len(prompts) for preds in predictions):
        raise ValueError("Incomplete predictions")
    tasks = list(zip(prompts, fresh, zip(*predictions)))
    chunk_size = (len(tasks)+workers-1)//workers
    args = [(cfg,tasks[i:i+chunk_size]) for i in range(0,len(tasks),chunk_size)]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scored = [r for part in pool.map(score_chunk,args) for r in part]
    for target in (*TARGETS,"observed"):
        rows = [dict(pair_id=r["pair_id"],source_id=r["source_id"],audit_split=r["audit_split"],**r[target]) for r in scored]
        write_jsonl(out / f"{target}.jsonl",rows)
    print(f"Scored {len(scored)} fresh completions in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("predict","score"))
    parser.add_argument("--run",type=Path,default=Path("artifacts/m7vv_q4"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/m7vv_q4_diagnostics/s_double_prime"))
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage == "predict":
        predict(args.run,args.out)
    else:
        score(args.run,args.out,args.workers)
