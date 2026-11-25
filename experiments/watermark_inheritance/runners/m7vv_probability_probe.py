"""Construction-only comparison of base and final A under the original key; no fitting."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, load_config, read_jsonl
from experiments.watermark_inheritance.score import eligible_predictions

RUN = Path("artifacts/m7vv_q4")
OUT = Path("artifacts/m7vv_q4_diagnostics/probability_probe")

@torch.inference_mode()
def probe():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(RUN)
    tok = AutoTokenizer.from_pretrained(RUN / "base", local_files_only=True)
    wm = Maryland(tok, config)
    rows = [row for row in read_jsonl(RUN / "prepared/pairs.jsonl") if row["audit_split"]=="calibration"]
    assert len(rows) == 400
    cases = [(row["pair_id"], side, row["prompt_ids"], row[side+"_ids"])
             for row in rows for side in ("s", "s_prime")]
    measurements = []
    model_paths = {"base": RUN / "base", "A": RUN / "train/A/checkpoints/target_epoch1"}
    for name, path in model_paths.items():
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to("cuda").eval()
        for offset in range(0,len(cases),8):
            batch = cases[offset:offset+8]
            width = max(len(p)+len(c) for _,_,p,c in batch)
            ids = torch.full((len(batch),width),tok.eos_token_id,device="cuda",dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i,(_,_,p,c) in enumerate(batch):
                ids[i,:len(p)+len(c)] = torch.tensor(p+c,device="cuda")
                mask[i,:len(p)+len(c)] = 1
            output = model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            for i,(pair, side,p,c) in enumerate(batch):
                x = output[i,len(p)-1:len(p)+len(c)-1,:len(tok)].float()
                lp = x.log_softmax(-1)
                target = torch.tensor(c,device="cuda")
                actual_lp = lp.gather(1,target[:,None])[:,0]
                argmax = lp.argmax(-1)
                # At most eight evenly spaced eligible positions per example; all examples equally weighted.
                eligible = list(eligible_predictions(p,c,list(range(len(c))),ngram=wm.ngram))
                n = min(8,len(eligible))
                selected = [eligible[j] for j in np.linspace(0,len(eligible)-1,num=n,dtype=int)]
                mass = float("nan")
                if n:
                    pos = torch.tensor([pos for _,pos in selected],device="cuda")
                    green = torch.stack([wm.greenlist(ctx) for ctx,_ in selected]).cuda()
                    mass = lp[pos].gather(1,green).exp().sum(1).mean().item()
                measurements.append(dict(model=name,pair_id=pair,side=side,tokens=len(c),green_positions=n,
                    mean_logprob=actual_lp.mean().item(),min_k20=actual_lp.topk(math.ceil(.2*len(c)),largest=False).values.mean().item(),
                    top1_exact=(argmax==target).float().mean().item(),green_mass=mass))
        del model
    fields = ("mean_logprob","min_k20","top1_exact","green_mass")
    with (OUT / "rows.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=measurements[0]); w.writeheader(); w.writerows(measurements)
    summary={}
    for name in model_paths:
        summary[name]={}
        for side in ("s","s_prime"):
            rs=[r for r in measurements if r["model"]==name and r["side"]==side]
            summary[name][side]={k:float(np.nanmean([r[k] for r in rs])) for k in fields}
        summary[name]["s_minus_s_prime"]={k:summary[name]["s"][k]-summary[name]["s_prime"][k] for k in fields}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    torch.set_num_threads(1)
    probe()
