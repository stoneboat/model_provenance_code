"""Verify the completed run against frozen inputs and build the four-checkpoint comparison."""
import csv
import json
import math
from pathlib import Path

report = Path(__file__).resolve().parent
root = report.parents[3]
out = root / "artifacts/gpt_neo_lamini"
prior = root / "artifacts/lamini_matched"
original = root / "artifacts/pile_controls_reconstruction/20261124T160154122852Z"
key_names = ("min_k_20_logprob", "mean_logprob")

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

neo = original / "independent_EleutherAI__gpt-neo-1.3B"
sources = [
    ("Pythia-1.4B", original / "parent_EleutherAI__pythia-1.4b", prior / "witness_audits/pythia_base"),
    ("Pythia-1.4B + Lamini", prior / "pythia/mimir", prior / "witness_audits/pythia_adapted"),
    ("GPT-Neo-1.3B", neo, neo),
    ("GPT-Neo-1.3B + Lamini", out / "mimir", out / "audit"),
]
neo_shuffles = {r["score"]:float(r["shuffled_gap"]) for r in read_csv(
    report.parent / "2026-11-24-lamini-design/cached_audits.csv") if r["checkpoint"]=="GPT-Neo current split"}
frozen = root / "data/processed/mimir_github"
ref = math.sqrt(math.log(2/0.05)/200)
adj = math.sqrt(math.log(4/0.05)/200)
table = []
for name, score_dir, audit_dir in sources:
    result = read(audit_dir / "results.json")
    shuffled = {r["score_name"]:r["test_advantage"] for r in result["shuffled_label_control"] or []}
    for split in ("train","test"):
        raw = rows(frozen / (split + ".jsonl"))
        scored = rows(score_dir / (split + "_scores.jsonl"))
        assert len(raw)==len(scored) and all(all(a[k]==b[k] for k in ("id","label","phase_split","text_hash")) for a,b in zip(raw,scored))
        if score_dir == out / "mimir":
            baseline = rows(neo / (split + "_scores.jsonl"))
            assert all(all(a[k]==b[k] for k in ("num_input_tokens","num_scored_tokens")) for a,b in zip(scored,baseline))
    for key in key_names:
        r = next(row for row in result["main_results"] if row["score_name"]==key)
        threshold = r["calibrated_threshold"]
        accepted = [sum(z["label"]==label and z[key]>=threshold for z in scored) for label in (1,0)]
        gap = (accepted[0]-accepted[1])/200
        t = r["test"]
        assert (t["n_pos"],t["n_neg"],t["tp"],t["fp"],t["shard_advantage"])==(200,200,*accepted,gap)
        shuffle = neo_shuffles[key] if audit_dir==neo else shuffled[key]
        table.append(dict(checkpoint=name, score=key, fitted_threshold=threshold, member_accepted=accepted[0],
                          nonmember_accepted=accepted[1], held_out_per_source=200, signed_advantage=gap,
                          absolute_advantage=abs(gap), individual_cutoff=ref, reference_detection=gap>ref,
                          two_score_cutoff=adj, above_two_score_cutoff=gap>adj, shuffled_advantage=round(shuffle,6)))
with (report / "comparison.csv").open("w",newline="") as f:
    writer = csv.DictWriter(f,fieldnames=table[0],lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
checkpoint = out / "checkpoint"
train = read(checkpoint / "training.json")
order = read(checkpoint / "row_order.json")
assert order == read(prior / "pythia/row_order.json") and len(order)==3 and all(sorted(e)==list(range(1260)) for e in order)
trace = rows(checkpoint / "training_trace.jsonl")
assert len(trace)==train["trained_steps"]==237
assert all(e["examples"]==1260 and e["supervised_tokens"]==78293 for e in train["epochs"])
assert read(checkpoint / "effective_train_config.json")["fine_tuning"] == read(prior / "pythia/effective_train_config.json")["fine_tuning"]
assert len(read(checkpoint / "zero_delta.json"))==4 and all(e["zero_delta"]==e["base"] for e in read(checkpoint / "zero_delta.json"))
assert "Falling back" not in (out / "logs/scoring.log").read_text()
adapt = read(out / "downstream/adaptation.json")["metrics"]
assert adapt["test"]["n_rows"]==140 and adapt["test"]["answer_eos_tokens"]==8576 and adapt["test_unseen"]["n_rows"]==124
print("Verified all four checkpoints, exact training row coverage, per-example scores, zero-update export, and downstream evaluation.")
