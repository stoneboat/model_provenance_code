"""Independently replay the Q4 key on stored tokens and cached target predictions, on CPU."""
import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.watermark_inheritance.common import fingerprint, read_jsonl, write_json
from experiments.watermark_inheritance.generate import watermark_config_id
from experiments.watermark_inheritance.train import response_example

RUN = Path("artifacts/m7vv_q4")
OUT = Path("artifacts/m7vv_q4_diagnostics/key_checks")
SIDES = ("s", "s_prime", "s_double_prime")
TARGETS = ("A","B","C")

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def make_index(config, vocab):
    pairs = read_jsonl(RUN / 'prepared/pairs.jsonl')
    fresh = read_jsonl(RUN / 'prepared/s_double_prime.jsonl')
    pred = {name:read_jsonl(RUN / f'scores/{name}/predictions.jsonl') for name in TARGETS}
    eligible = []
    row_meta = []
    for i,(p,f) in enumerate(zip(pairs,fresh,strict=True)):
        assert p['pair_id']==f['pair_id'] and p['audit_split']==f['audit_split']
        assert p['watermark_config_id']==f['watermark_config_id']==watermark_config_id(config)
        prompt = p['prompt_ids']
        completions = (p['s_ids'],p['s_prime_ids'],f['completion_ids'])
        for side,tokens in zip(SIDES,completions,strict=True):
            rowid = len(row_meta)
            row_meta.append(dict(pair_id=p['pair_id'],audit_split=p['audit_split'],shard=side))
            history = prompt + tokens
            assert min(history)>=0 and max(history)<vocab and len(prompt)==64 and len(tokens)<=256
            example = response_example(prompt,tokens,512)
            assert example['labels']==[-100]*len(prompt)+tokens and example['input_ids']==history
            predictions = []
            for name in TARGETS:
                record = pred[name][i]
                assert record['pair_id']==p['pair_id']
                if side!='s_double_prime':
                    predictions.append(record[side]['predictions'])
                    assert len(predictions[-1])==len(tokens) and max(predictions[-1])<vocab
                else:
                    predictions.append(tokens)
            seen = set(zip(prompt[:-2],prompt[1:-1]))
            for j,token in enumerate(tokens):
                start = len(prompt)+j-2
                context = (history[start],history[start+1])
                if context in seen:
                    continue
                seen.add(context)
                eligible.append((context[0]*vocab+context[1],rowid,token,*(seq[j] for seq in predictions)))
    array = np.asarray(eligible,dtype=np.int64)
    array = array[np.argsort(array[:,0],kind='stable')]
    return array,row_meta

def score_range(bounds):
    lo,hi = bounds
    records = observations[lo:hi]
    cuts = np.r_[0,np.flatnonzero(np.diff(records[:,0]))+1,len(records)]
    labels = np.zeros((len(records),5),dtype=np.uint8)
    mask = np.zeros(vocab,dtype=bool)
    rng = torch.Generator(device='cpu')
    for start,stop in zip(cuts[:-1],cuts[1:]):
        a,b=divmod(int(records[start,0]),vocab)
        for j,key in enumerate((cfg['watermark_key'],cfg['wrong_key'])):
            seed=((key*salt+a)*salt+b)%(2**64-1)
            rng.manual_seed(seed)
            green=torch.randperm(vocab,generator=rng)[:green_size].numpy()
            mask[green]=True
            if j==0:
                labels[start:stop,:4]=mask[records[start:stop,2:6]]
            else:
                labels[start:stop,4]=mask[records[start:stop,2]]
            mask[green]=False
    return np.stack([np.bincount(records[:,1],weights=lbl,minlength=n_rows).astype(int)
                     for lbl in labels.T])

if __name__=='__main__':
    torch.set_num_threads(1)
    cfg=json.loads((RUN/'config.json').read_text())
    tok=AutoTokenizer.from_pretrained(RUN/'base',local_files_only=True)
    vocab=len(tok)
    salt=cfg['salt_key']
    green_size=int(vocab*cfg['gamma'])
    gamma=green_size/vocab
    assert cfg['ngram']==2
    tokenizer_files = ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    # Saving a target tokenizer adds a pad token alias; directly compare the ID mapping.
    for t in TARGETS:
        target_tok=AutoTokenizer.from_pretrained(RUN/f'train/{t}/checkpoints/target_epoch1',local_files_only=True)
        assert tok.get_vocab()==target_tok.get_vocab()
        meta=json.loads((RUN/f'scores/{t}/predictions.meta.json').read_text())
        assert meta['pairs_sha256']==digest(RUN/'prepared/pairs.jsonl') and not meta['watermark_bias']
        assert meta['config_sha256']==fingerprint(cfg)
        assert meta['scorer_sha256']==digest(RUN/f'scores/{t}/scoring_source.py')
    observations,meta=make_index(cfg,vocab)
    n_rows=len(meta)
    groups=np.r_[0,np.flatnonzero(np.diff(observations[:,0]))+1,len(observations)]
    print(f"{len(groups)-1} unique contexts over {len(observations)} eligible positions",flush=True)
    boundaries=groups[np.linspace(0,len(groups)-1,6,dtype=int)]
    with mp.get_context('fork').Pool(5) as pool:
        counts=np.sum(pool.map(score_range,list(zip(boundaries[:-1],boundaries[1:]))),axis=0)
    n=np.bincount(observations[:,1],minlength=n_rows)
    empirical=[]
    for idx,row in enumerate(meta):
        count=int(n[idx])
        sd=math.sqrt(count*gamma*(1-gamma))
        for method,col in [('correct',0),('wrong',4)]:
            green=int(counts[col,idx])
            empirical.append(dict(**row,key=method,eligible=count,green=green,z=(green-count*gamma)/sd if sd else 0.))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'emitted_scores.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,empirical[0].keys());w.writeheader();w.writerows(empirical)
    for t,col in zip(TARGETS,[1,2,3]):
        for i,r in enumerate(read_jsonl(RUN/f'scores/{t}/examples.jsonl')):
            for side_idx,side in enumerate(SIDES[:2]):
                j=i*3+side_idx
                assert r['pair_id']==meta[j]['pair_id'] and r[side]['green']==counts[col,j] and r[side]['eligible']==n[j]
                expected=(counts[col,j]-n[j]*gamma)/math.sqrt(n[j]*gamma*(1-gamma)) if n[j] else 0.
                assert abs(r[side]['z']-expected)<1e-12
    summary=[]
    for split in ('calibration','audit','all'):
        for k in ('correct','wrong'):
            for side in SIDES:
                rows=[r for r in empirical if (split=='all' or r['audit_split']==split) and r['shard']==side and r['key']==k]
                total=sum(r['eligible'] for r in rows)
                green=sum(r['green'] for r in rows)
                z=[r['z'] for r in rows]
                summary.append(dict(split=split,key=k,shard=side,rows=len(rows),eligible=total,green=green,
                                    green_rate=green/total,mean_z=np.mean(z),sd_z=np.std(z,ddof=1)))
    report=Path("experiments/watermark_inheritance/results/2026-11-25-m7vv-diagnostics")
    with (report/'emitted_watermark.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,summary[0].keys());w.writeheader();w.writerows(summary)
    write_json(report/'implementation_checks.json',dict(vocab_size=vocab,model_vocab=json.loads((RUN/'base/config.json').read_text())['vocab_size'],
        gamma_exact=gamma,green_size=green_size,ngram=cfg['ngram'],salt=salt,key=cfg['watermark_key'],wrong_key=cfg['wrong_key'],
        z_replays_verified=2000*2*3,generated_rows=len(meta),eligible_positions=len(observations),
        unique_contexts=len(groups)-1,token_to_id_map_identical_all_final_targets=True,
        original_generation_and_scoring_config_match=True,training_response_mask_alignment_verified=True,
        scoring_shift='position len(prompt)-1+i predicts stored completion i',original_reading_score_bias=False))
