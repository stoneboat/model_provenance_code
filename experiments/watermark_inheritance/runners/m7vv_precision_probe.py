"""Precision diagnostic for A on construction pairs only, with the original keyed readout."""
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl
from experiments.watermark_inheritance.score import eligible_predictions

RUN = Path("artifacts/m7vv_q4")
OUT = Path("artifacts/m7vv_q4_diagnostics/precision_probe")
PATH = RUN / "train/A/checkpoints/target_epoch1"

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True,exist_ok=True)
    tok = AutoTokenizer.from_pretrained(RUN / "base",local_files_only=True)
    rows = [r for r in read_jsonl(RUN/"prepared/pairs.jsonl") if r["audit_split"]=="calibration"]
    assert len(rows)==400
    cases=[(r["pair_id"],side,r["prompt_ids"],r[side+"_ids"]) for r in rows for side in ("s","s_prime")]
    saved = {r["pair_id"]:r for r in read_jsonl(RUN/"scores/A/predictions.jsonl")}
    wm = Maryland(tok,load_config(RUN))
    modes = ["bf16","fp32_head","fp32"]
    records = {mode:[] for mode in modes}
    predictions = {mode:{} for mode in modes}
    for dtype in (torch.bfloat16,torch.float32):
        model = AutoModelForCausalLM.from_pretrained(PATH,torch_dtype=dtype,attn_implementation="sdpa",local_files_only=True).cuda().eval()
        for begin in range(0,len(cases),8):
            batch=cases[begin:begin+8]
            width=max(len(p)+len(c) for _,_,p,c in batch)
            inputs=torch.full((len(batch),width),tok.eos_token_id,device="cuda",dtype=torch.long)
            mask=torch.zeros_like(inputs)
            for i,(_,_,p,c) in enumerate(batch):
                inputs[i,:len(p)+len(c)]=torch.tensor(p+c,device="cuda")
                mask[i,:len(p)+len(c)]=1
            hidden=model.gpt_neox(input_ids=inputs,attention_mask=mask,use_cache=False).last_hidden_state
            for mode in (["bf16","fp32_head"] if dtype==torch.bfloat16 else ["fp32"]):
                logits=(model.embed_out(hidden) if mode!="fp32_head" else
                        torch.nn.functional.linear(hidden.float(),model.embed_out.weight.float()))
                for i,(pair,side,p,c) in enumerate(batch):
                    x=logits[i,len(p)-1:len(p)+len(c)-1,:len(tok)].float()
                    top,arg=x.max(1)
                    refs=torch.tensor(saved[pair][side]["predictions"],device="cuda")
                    lp=x.log_softmax(1).gather(1,torch.tensor(c,device="cuda")[:,None])[:,0]
                    tied=x.eq(top[:,None]).sum(1)
                    predictions[mode][pair,side]=arg.tolist()
                    records[mode].append(dict(mode=mode,pair_id=pair,side=side,n=len(c),
                        mean_top_logit=top.mean().item(),mean_abs_top_logit=top.abs().mean().item(),
                        top_tie_fraction=(tied>1).float().mean().item(),mean_winners=tied.float().mean().item(),
                        cached_top1_agreement=(arg==refs).float().mean().item(),
                        min_k20=lp.topk(math.ceil(.2*len(c)),largest=False).values.mean().item(),mean_lp=lp.mean().item()))
                del logits
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for i,(pair,side,p,c) in enumerate(cases):
        eligible=list(eligible_predictions(p,c,range(len(c)),wm.ngram))
        green={k:0 for k in modes}
        for ctx,pos in eligible:
            for mode in modes:
                green[mode]+=wm.is_green(ctx,predictions[mode][pair,side][pos])
        for mode in modes:
            n=len(eligible); g=wm.gamma_exact
            records[mode][i]["z"]=(green[mode]-n*g)/math.sqrt(n*g*(1-g)) if n else 0.
    with (OUT/"rows.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=records["bf16"][0]);w.writeheader()
        for mode in modes: w.writerows(records[mode])
    fields=["z","top_tie_fraction","mean_winners","cached_top1_agreement","mean_abs_top_logit","mean_top_logit","mean_lp","min_k20"]
    summary={}
    for mode in modes:
        summary[mode]={}
        for side in ("s","s_prime"):
            selected=[row for row in records[mode] if row["side"]==side]
            summary[mode][side]={key:float(np.mean([row[key] for row in selected])) for key in fields}
        xs=[r["z"] for r in records[mode] if r["side"]=="s"]
        ys=[r["z"] for r in records[mode] if r["side"]=="s_prime"]
        summary[mode]["construction_auc"]=(rankdata(xs+ys)[:400].sum()-400*401/2)/400**2
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
