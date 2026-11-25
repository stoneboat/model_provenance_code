"""Verify that the training projections implement the frozen three-shard design."""
from collections import Counter
from pathlib import Path

from .common import load_config, read_jsonl, write_json
from .generate import validate_pair
from .null_control import validate_double_prime
from .prepare import file_sha256

def verify(run):
    cfg = load_config(run)
    folder = run / "prepared"
    pairs = read_jsonl(folder / "pairs.jsonl")
    prompts = read_jsonl(folder / "prompts.jsonl")
    third = read_jsonl(folder / "s_double_prime.jsonl")
    train_s = read_jsonl(folder / "s_train.jsonl")
    train_third = read_jsonl(folder / "s_double_prime_train.jsonl")
    assert set(map(len,[pairs,prompts,third,train_s,train_third]))=={2000}
    assert Counter(r["audit_split"] for r in prompts)=={"calibration":400,"audit":1600}
    assert len({r["article_id"] for r in prompts})==2000
    pilot = read_jsonl(folder / "pilot_prompts.jsonl")
    assert not {p["article_id"] for p in pilot} & {p["article_id"] for p in pairs}
    n_ties=[0,0,0]
    for p,r,z,s,t in zip(prompts,pairs,third,train_s,train_third):
        validate_pair(r,p,cfg,"full")
        validate_double_prime(z,r,cfg)
        common = dict(pair_id=p["pair_id"],prompt_ids=p["prompt_ids"])
        assert s == dict(**common,s_ids=r["s_ids"])
        assert t == dict(**common,s_ids=z["completion_ids"])
        n_ties[0] += s["s_ids"]==r["s_prime_ids"]
        n_ties[1] += s["s_ids"]==t["s_ids"]
        n_ties[2] += t["s_ids"]==r["s_prime_ids"]
    seeds = {r["sampling_seed"] for r in third} | {p[k] for p in pairs for k in ("sampling_seed_s","sampling_seed_s_prime")}
    assert len(seeds)==6000
    summary = dict(pairs=2000, calibration_pairs=400, evaluation_pairs=1600, distinct_articles=2000,
                   distinct_sampling_seeds=6000, s_sprime_ties=n_ties[0], s_sdoubleprime_ties=n_ties[1],
                   sprime_sdoubleprime_ties=n_ties[2], exactly_projected_training_files=True,
                   projection_fields=["pair_id","prompt_ids","s_ids"],
                   s_training_file_sha256=file_sha256(folder / "s_train.jsonl"),
                   s_double_prime_training_file_sha256=file_sha256(folder / "s_double_prime_train.jsonl"))
    write_json(folder / "generation_ready.json", summary)
    print(summary,flush=True)

if __name__=="__main__":
    verify(Path("artifacts/m7vv_q4"))
