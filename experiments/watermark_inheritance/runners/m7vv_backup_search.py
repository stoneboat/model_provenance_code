"""Find the first higher-exposure construction-only candidate passing the preregistered two-fold gate."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.watermark_inheritance.common import load_config, read_jsonl, write_json, write_jsonl
from experiments.watermark_inheritance.runners import m7vv_construction as scoring
from experiments.watermark_inheritance.runners.minimal_q4_train import TEMPLATE, digest, run_stage

EPOCHS = [6, 9, 12, 18, 24]
FAMILIES = ["bf16_lr5", "head32_lr5"]
ROOT = Path("artifacts/m7vv_minimal/backup")
SEARCH_SEED = 34711


def score(target, rows, tokenizer):
    predictions = scoring.predict(rows, tokenizer, target)
    examples = scoring.apply_membership(rows, predictions)
    assert len(examples) == 400
    return examples, scoring.summarize(examples)


def record_result(rows, tokenizer, path, family, epoch, root):
    run = root / family
    checkpoint_name = "parent" if epoch == 24 else f"parent_epoch{epoch}"
    witness = run / "checkpoints" / checkpoint_name
    dolly = run / f"dolly_epoch{epoch}"
    target = dolly/"checkpoints/target_epoch1"
    ledger = json.loads((dolly / f"ledger_target_epoch{epoch}.json").read_text())
    assert ledger["source_sha256"] == digest(witness / "model.safetensors")
    assert ledger["weights_sha256"] == digest(target / "model.safetensors")
    examples, metrics = score(target, rows, tokenizer)
    primary = metrics["z"]
    gaps = [entry["gap"] for entry in primary["crossfit"]]
    passed = min(gaps) >= .15 - 1e-12 and primary["twofold_gap"] >= .20 - 1e-12
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path.parent/"scores.jsonl", examples)
    result = dict(family=family, epoch=epoch, construction_pairs=len(examples),
                  passes_backup_gate=bool(passed), metrics=metrics,
                  witness_checkpoint=str(witness), final_checkpoint=str(target),
                  parent_sha256=ledger["source_sha256"], target_sha256=ledger["weights_sha256"],
                  scored_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(path,result)
    print(json.dumps(dict(epoch=epoch,family=family,fold_gaps=gaps,mean=primary["twofold_gap"],passes=bool(passed))),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--through",type=int,choices=EPOCHS,default=max(EPOCHS))
    args = parser.parse_args()
    root = args.root
    rows = read_jsonl(scoring.ORIGINAL/"prepared/pairs.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(load_config(scoring.ORIGINAL)["tokenizer_path"],local_files_only=True)
    for epoch in EPOCHS:
        if epoch > args.through:
            break
        for family in FAMILIES:
            run = root / family
            result_path = root / "construction" / f"{family}_e{epoch}" / "summary.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
            else:
                base=dict(family=family, seed=SEARCH_SEED, root=root, witness=TEMPLATE/"prepared/pairs.jsonl",
                          epoch=epoch, checkpoint_epochs=EPOCHS, save_only_requested=False)
                tick=time.monotonic()
                run_stage(argparse.Namespace(stage="parent",**base))
                run_stage(argparse.Namespace(stage="target",**base))
                torch.set_num_threads(1)
                result = record_result(rows,tokenizer,result_path,family,epoch,root)
                print(f"Candidate wall seconds: {time.monotonic()-tick:.1f}",flush=True)
                # A failed final branch is completely represented by its digest and scores.
                if not result["passes_backup_gate"]:
                    (Path(result["final_checkpoint"])/"model.safetensors").unlink()
                # The new resume checkpoint supersedes the preceding failed prefix of this family.
                earlier = [e for e in EPOCHS if e < epoch]
                if earlier:
                    old=run/"checkpoints"/f"parent_epoch{earlier[-1]}"/"model.safetensors"
                    old.unlink(missing_ok=True)
            if result["passes_backup_gate"]:
                write_json(root/"selected.json",result)
                return
    if args.through == max(EPOCHS):
        write_json(root/"no_selection.json",dict(families=FAMILIES,epochs=EPOCHS,passed=False))


if __name__ == "__main__":
    main()
