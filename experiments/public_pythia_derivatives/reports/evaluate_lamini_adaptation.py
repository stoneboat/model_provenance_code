"""Measure base-to-adapter changes in answer-token-weighted downstream loss."""
import argparse
import csv
import gc
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runners"))
import lamini_data
from lamini_data import collate_rows

DESIGN = HERE.parent / "results/2026-11-24-lamini-design"
BOOTSTRAP_SEED = 3324
BOOTSTRAP_DRAWS = 10000


def normalized_words(text):
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def unseen_test_rows(train, test):
    questions = {normalized_words(row["question"]) for row in train}
    answers = {normalized_words(row["answer"]) for row in train}
    return np.array([normalized_words(row["question"]) not in questions and
                     normalized_words(row["answer"]) not in answers for row in test])


@torch.inference_mode()
def evaluate_split(model, encoded, pad_token_id, batch_size, device):
    sums = []
    counts = []
    for offset in range(0, len(encoded), batch_size):
        tensors = collate_rows(encoded[offset:offset + batch_size], pad_token_id)
        labels = tensors.pop("labels").to(device)[:, 1:]
        inputs = {key: tensor.to(device) for key, tensor in tensors.items()}
        logits = model(**inputs, use_cache=False).logits[:, :-1]
        # Select only supervised positions before casting logits to float32.
        selected = labels.ne(-100)
        token_losses = F.cross_entropy(logits[selected].float(), labels[selected], reduction="none")
        positions = torch.zeros_like(labels, dtype=torch.float32)
        positions[selected] = token_losses
        sums.extend(positions.sum(dim=1, dtype=torch.float64).cpu().tolist())
        counts.extend(selected.sum(dim=1).cpu().tolist())
    losses = np.array(sums, dtype=np.float64)
    counts = np.array(counts, dtype=np.int64)
    if not np.isfinite(losses).all() or not (counts > 0).all():
        raise ValueError("Invalid per-row answer loss or token count")
    return losses, counts


def paired_summary(before, after, counts, with_interval=False):
    pre = float(before.sum() / counts.sum())
    post = float(after.sum() / counts.sum())
    result = {
        "n_rows": len(counts),
        "answer_eos_tokens": int(counts.sum()),
        "pre_nll": pre,
        "post_nll": post,
        "pre_perplexity": math.exp(pre),
        "post_perplexity": math.exp(post),
        "nll_change_post_minus_pre": post - pre,
        "relative_nll_reduction": (pre - post) / pre,
    }
    if with_interval:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        index = rng.integers(len(counts), size=(BOOTSTRAP_DRAWS, len(counts)))
        change = (after - before)[index].sum(axis=1) / counts[index].sum(axis=1)
        result["nll_change_percentile_95"] = np.quantile(change, [0.025, 0.975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pythia", "biogpt", "gpt_neo"], required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Optional directory containing the pinned downstream release")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", type=Path, default=DESIGN / "protocol.yaml")
    args = parser.parse_args()
    start = time.monotonic()
    protocol = yaml.safe_load(args.protocol.read_text())
    base = protocol["bases"][args.model]
    # The published mixed-case BioGPT ID resolves to the lowercase cached repository.
    base_id = "microsoft/biogpt-large" if args.model == "biogpt" else base["model"]
    revision = base["revision"]
    tok = AutoTokenizer.from_pretrained(base_id, revision=revision)
    if type(tok).__name__ != base["tokenizer_class"] or tok.padding_side != "right":
        raise ValueError("Unexpected native tokenizer or padding side")
    pad_id = tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id
    data = lamini_data
    is_dolly = protocol["training_data"]["dataset"] == "databricks/databricks-dolly-15k"
    if is_dolly:
        import dolly_data as data
        splits = ("train",)
        data_dir = args.data_dir or Path("artifacts/dolly_matched/data")
        preflight = json.loads((data_dir / f"{args.model}_preflight.json").read_text())
        expected_targets = {"train": preflight["supervised_tokens"]}
    else:
        splits = ("train", "test")
        data_dir = args.data_dir or Path("artifacts/lamini_design")
        design_counts = json.loads((args.protocol.parent / "tokenization.json").read_text())[args.model]
        expected_targets = {s: design_counts[s]["separate_answer_plus_eos"]["total_tokens"] for s in splits}
    rows = {s: data.load_split(data_dir, s) for s in splits}
    unseen = None
    if "test" in rows:
        unseen = unseen_test_rows(rows["train"], rows["test"])
        if unseen.sum() != 124:
            raise ValueError(f"Expected 124 test rows with previously unseen question and answer, got {unseen.sum()}")
    encoded = {s: [data.encode_row(row, tok) for row in rows[s]] for s in splits}
    device = torch.device(args.device)
    scores = {}
    token_counts = {}
    for stage, model_id, rev in [("pre", base_id, revision), ("post", str(args.merged), None)]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, torch_dtype=torch.float16, device_map={"": device}
        )
        model.eval()
        for split in splits:
            losses, n = evaluate_split(model, encoded[split], pad_id, args.batch_size, device)
            if int(n.sum()) != expected_targets[split]:
                raise ValueError(f"Answer/EOS tokens differ from preflight for {split}")
            if split in token_counts and not np.array_equal(token_counts[split], n):
                raise ValueError(f"Base and tuned token counts differ in split {split}")
            scores[stage, split] = losses
            token_counts[split] = n
            print(f"{args.model} {stage} {split} NLL {losses.sum()/n.sum():.6f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "row_losses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["split", "row_index", "both_fields_unseen_in_train",
                         "answer_eos_tokens", "pre_nll_sum", "post_nll_sum"])
        for split in splits:
            for i, n in enumerate(token_counts[split]):
                writer.writerow([split, i, bool(unseen[i]) if split == "test" else "",
                                 n, scores["pre", split][i], scores["post", split][i]])

    metrics = {s: paired_summary(scores["pre", s], scores["post", s], token_counts[s],
                                 with_interval=(s == "test")) for s in splits}
    if unseen is not None:
        metrics["test_unseen"] = paired_summary(scores["pre", "test"][unseen],
                                                scores["post", "test"][unseen],
                                                token_counts["test"][unseen])
    report = {
        "base": base_id, "base_revision": revision, "merged": str(args.merged),
        "dataset_revision": protocol["training_data"]["revision"],
        "tokenizer": type(tok).__name__, "compute_dtype": "torch.float16",
        "cross_entropy_dtype": "torch.float32", "row_summation_dtype": "torch.float64",
        "device": str(device), "batch_size": args.batch_size,
        "bootstrap_draws": BOOTSTRAP_DRAWS if unseen is not None else 0,
        "bootstrap_seed": BOOTSTRAP_SEED if unseen is not None else None,
        "wall_seconds": time.monotonic() - start, "metrics": metrics,
    }
    (args.output_dir / "adaptation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
