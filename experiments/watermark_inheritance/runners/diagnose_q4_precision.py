"""Diagnostic of BF16 rounding on a fixed prefix of Q4 construction pairs, using CPU only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import read_jsonl, write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",choices=("A","base"),default="A")
parser.add_argument("--pairs",type=int,default=16)
args=parser.parse_args()
root=Path("artifacts/m7vv_q4")
report=Path("experiments/watermark_inheritance/results/2026-11-25-m7vv-diagnostics")
torch.set_num_threads(2)
rows=read_jsonl(root/'prepared/pairs.jsonl')
pred=read_jsonl(root/'scores/A/predictions.jsonl')
samples=[(row,stored) for row,stored in zip(rows,pred,strict=True) if row['audit_split']=='calibration'][:args.pairs]
tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True)
checkpoint=root/'train/A/checkpoints/target_epoch1' if args.model=='A' else root/'base'
met={}
predictions={}
details={}
with torch.inference_mode():
    for dtype in [torch.float32,torch.bfloat16]:
        net=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=dtype,attn_implementation='sdpa').eval()
        fields={key:[] for key in ['row_logit_mean','max_logit','median_logit','vocab_maxima_ties','top_margin','ulp_at_max','matches_cached']}
        y=[]
        for row,stored in samples:
            for side in ('s','s_prime'):
                prompt=row['prompt_ids']
                completion=row[side+'_ids']
                ids=torch.tensor([prompt+completion])
                outputs=net(input_ids=ids,use_cache=False)
                logits=outputs.logits[0,len(prompt)-1:-1,:len(tok)]
                lp=logits.float()
                top=lp.topk(2,dim=-1).values
                argmax=lp.argmax(dim=-1)
                y.extend(argmax.tolist())
                peak=logits.max(dim=-1).values
                ulp=(torch.nextafter(peak,torch.full_like(peak,torch.inf))-peak).float()
                data=[lp.mean(dim=-1),peak.float(),lp.median(dim=-1).values,(lp==peak[:,None]).sum(dim=-1),
                      top[:,0]-top[:,1],ulp,argmax==torch.tensor(stored[side]['predictions'])]
                for key,val in zip(fields,data):
                    fields[key].extend(val.tolist())
        met[str(dtype)]={k:dict(mean=float(np.mean(v)),q0=float(np.min(v)),median=float(np.median(v)),
                                     q95=float(np.quantile(np.asarray(v,dtype=float),.95)),q1=float(np.max(v))) for k,v in fields.items()}
        met[str(dtype)]['any_top_tie_rate']=float(np.mean(np.array(fields['vocab_maxima_ties'])>1))
        if args.model=='base':
            del met[str(dtype)]['matches_cached']
        predictions[str(dtype)]=np.array(y)
        del net
name='precision_check.json' if args.model=='A' else 'base_precision_check.json'
write_json(report/name,dict(n_pairs=args.pairs,split='first construction pairs, both sides, all completion positions',
    target=args.model,device='CPU',backend='SDPA',n_tokens=len(y),measures=met,
    fp32_bf16_top1_agreement=float(np.mean(predictions['torch.bfloat16']==predictions['torch.float32']))))
