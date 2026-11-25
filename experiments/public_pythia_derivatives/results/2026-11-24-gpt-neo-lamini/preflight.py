"""Freeze the inherited protocol, verify row identity, and smoke-test one disposable LoRA update."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / "runners"))
import lamini_data
import train_matched_lamini as runner

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def save(name, value):
    (here / name).write_text(json.dumps(value, indent=2) + "\n")

rev = "dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8"
model = "EleutherAI/gpt-neo-1.3B"
base = dict(model=model, revision=rev, tokenizer_class="GPT2TokenizerFast",
            lora_module_suffixes=["attn.attention.out_proj", "mlp.c_proj"],
            expected_modules=48, trainable_parameters=5505024)
prior = Path("artifacts/lamini_matched/pythia")
cfg = read(prior / "effective_train_config.json")
cfg.update(status="fixed_before_training", question="Does the MIMIR GitHub witness detect independently pretrained GPT-Neo after the matched Lamini adaptation?", bases={"gpt_neo": base})
audit = cfg["witness_audit"]
audit.update(primary_family="GPT-Neo-Lamini is the single new primary MIN-K20 test. Also show the original two-arm numerical reference for context.", secondary_family="Mean log probability is secondary. Also compare both metrics against the two-score Bonferroni reference.", prior_checkpoints="Three cached comparators are descriptive.", descriptive_changes="Report tuned minus base within GPT-Neo; thresholds fitted separately.")
sources = [Path(runner.BASE_SCORES["pythia"]), prior / "mimir", Path(runner.BASE_SCORES["gpt_neo"])]
for s in sources:
    m = read(s / "manifest.json")
    assert (m["dtype"], m["max_length"], m["batch_size"], m["quantization"]) == ("torch.float16",512,4,"none") and 20 in m["k_pcts"]
    if s == sources[-1]:
        assert (m["model_revision"], m["tokenizer_revision"], m["model"], m["tokenizer_model"]) == (rev, rev, model, model)
    for split,count in [("train",500),("test",200)]:
        raw = rows(Path(audit["inputs"]) / (split+".jsonl"))
        cached = rows(s / (split+"_scores.jsonl"))
        assert Counter(x["label"] for x in raw) == {0:count,1:count}
        assert len(raw)==len(cached) and all(all(a[k]==b[k] for k in ("id","label","text_hash","phase_split")) for a,b in zip(raw,cached))
tok = AutoTokenizer.from_pretrained(model, revision=rev, local_files_only=True)
assert type(tok).__name__ == base["tokenizer_class"]
counts = {}
for split in ("train","test"):
    enc = [lamini_data.encode_row(r,tok) for r in lamini_data.load_split(Path("artifacts/lamini_design"),split)]
    count = sum(sum(t != -100 for t in labels[1:]) for _,labels in enc)
    counts[split] = {"separate_answer_plus_eos":{"total_tokens":count},"rows":len(enc)}
    if split=="train":
        tiny = enc[:4]
        cfg["training_data"]["expected_counts_per_epoch"] = {"gpt_neo":dict(input_tokens=sum(len(i) for i,l in enc),supervised_tokens=count,max_input_length=max(len(i) for i,l in enc))}
rng = np.random.default_rng(42)
assert [rng.permutation(1260).tolist() for _ in range(3)] == read(prior / "row_order.json")
save("tokenization.json", {"gpt_neo":counts})
(here / "protocol.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
torch.manual_seed(42)
net = runner.new_adapter(runner.load_base(base,torch.bfloat16,training=True),base,cfg["fine_tuning"])
net.train()
net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
batch = {k:v.cuda() for k,v in lamini_data.collate_rows(tiny,tok.eos_token_id).items()}
labels = batch.pop("labels")[:,1:]
active = labels != -100
with torch.autocast("cuda",dtype=torch.bfloat16):
    logits = net(**batch,use_cache=False).logits[:,:-1]
    loss = torch.nn.functional.cross_entropy(logits[active].float(),labels[active])
loss.backward()
params = [p for p in net.parameters() if p.requires_grad]
norm = torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
torch.optim.AdamW(params,lr=0.0002,weight_decay=0).step()
changed = sum(bool(torch.count_nonzero(p)) for n,p in net.named_parameters() if ".lora_B." in n)
assert torch.isfinite(loss) and changed == 48
save("smoke.json",dict(answer_nll=loss.item(),gradient_norm=float(norm),changed_B_matrices=changed,rows=4,verified_cached_score_directories=list(map(str,sources)),identical_3epoch_order=True))
