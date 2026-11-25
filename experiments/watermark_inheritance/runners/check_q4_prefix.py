"""Check archived top-1 logits against prefix-only reads and diagnose green votes on matches."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.watermark_inheritance.common import Maryland, read_jsonl, write_json
from experiments.watermark_inheritance.score import eligible_predictions

root=Path("artifacts/m7vv_q4")
report=Path("experiments/watermark_inheritance/results/2026-11-25-m7vv-diagnostics")
cfg=json.loads((root/'config.json').read_text())
torch.set_num_threads(1)
tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True)
wm=Maryland(tok,cfg)
pairs=read_jsonl(root/'prepared/pairs.jsonl')
pred=read_jsonl(root/'scores/A/predictions.jsonl')
construction=[(p,r) for p,r in zip(pairs,pred,strict=True) if p['audit_split']=='calibration']
assert len(construction)==400
net=AutoModelForCausalLM.from_pretrained(root/'train/A/checkpoints/target_epoch1',local_files_only=True,
                                         torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
prefix_rows=[]
for p,r in construction[:4]:
    assert p['pair_id']==r['pair_id']
    for side in ('s','s_prime'):
        completion=p[side+'_ids']
        for offset in sorted({0,min(1,len(completion)-1),min(7,len(completion)-1),len(completion)-1}):
            ids=torch.tensor([p['prompt_ids']+completion[:offset]])
            with torch.inference_mode():
                logits=net(input_ids=ids,use_cache=False).logits[0,-1,:len(tok)].float()
            cpu_pred=int(logits.argmax())
            saved=r[side]['predictions'][offset]
            prefix_rows.append(dict(pair_id=p['pair_id'],side=side,offset=offset,expected_input_length=ids.shape[1],
                                    cached_gpu_top1=saved,prefix_only_cpu_top1=cpu_pred,
                                    cpu_logit_deficit_of_cached=float(logits.max()-logits[saved])))
write_json(report/'prefix_checks.json',dict(checkpoint='A',precision='bfloat16',cpu_backend='sdpa',rows=prefix_rows,
    exact_top1_matches=sum(x['cached_gpu_top1']==x['prefix_only_cpu_top1'] for x in prefix_rows),n=len(prefix_rows)))

counts=defaultdict(lambda:[0,0,0])
for p,r in construction[:64]:
    for side in ('s','s_prime'):
        emitted=p[side+'_ids']
        predicted=r[side]['predictions']
        for (ctx,actual),(same_ctx,top1) in zip(eligible_predictions(p['prompt_ids'],emitted,emitted),
                                                 eligible_predictions(p['prompt_ids'],emitted,predicted),strict=True):
            assert ctx==same_ctx
            group=counts[side,actual==top1]
            group[0]+=1
            group[1]+=wm.is_green(ctx,actual)
            group[2]+=wm.is_green(ctx,top1)
with (report/'conditional_green_construction.csv').open('w',newline='') as stream:
    w=csv.writer(stream)
    w.writerow(['target','split','n_pairs','shard','top1_equals_observed','eligible','observed_green','predicted_green',
                'observed_green_rate','predicted_green_rate'])
    for (side,match),(n,actual,top1) in sorted(counts.items()):
        w.writerow(['A','first_64_construction',64,side,match,n,actual,top1,actual/n,top1/n])
