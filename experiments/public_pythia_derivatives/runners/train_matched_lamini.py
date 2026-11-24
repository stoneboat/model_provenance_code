"""Train and export one arm of a matched downstream experiment."""
import argparse
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import lamini_data
from lamini_data import collate_rows

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.shard_audit.scoring.logprobs import extract_token_logprobs
from src.shard_audit.scoring.mia_scores import compute_all_scores

DESIGN = ROOT / "experiments/public_pythia_derivatives/results/2026-11-24-lamini-design/protocol.yaml"
BASE_SCORES = {
    "pythia": "artifacts/pile_controls_reconstruction/20261124T160154122852Z/parent_EleutherAI__pythia-1.4b",
    "biogpt": "artifacts/biogpt_large_mimir",
}
DEVICE = torch.device("cuda:0")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_base(base, dtype, training=False):
    # HF's case-sensitive on-disk cache has the complete BioGPT snapshot under its lower-case ID.
    model_id = base["model"].replace("microsoft/BioGPT-Large", "microsoft/biogpt-large")
    source = dict(revision=base["revision"], local_files_only=True)
    config = AutoConfig.from_pretrained(model_id, **source)
    if training:
        for key in ("attention_dropout", "hidden_dropout", "attention_probs_dropout_prob",
                    "hidden_dropout_prob", "activation_dropout", "layerdrop"):
            if hasattr(config, key):
                setattr(config, key, 0.0)
        config.use_cache = False
    return AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=dtype, device_map={"": DEVICE}, **source
    )


def new_adapter(base_model, base, tuning):
    cfg = LoraConfig(
        task_type="CAUSAL_LM", r=tuning["rank"], lora_alpha=tuning["alpha"],
        lora_dropout=tuning["adapter_dropout"], target_modules=base["lora_module_suffixes"],
        init_lora_weights=True, bias="none", revision=base["revision"],
    )
    model = get_peft_model(base_model, cfg)
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if sum(p.numel() for _, p in params) != base["trainable_parameters"]:
        raise ValueError("Unexpected number of trainable adapter parameters")
    b_weights = [p for n, p in params if ".lora_B." in n]
    if len(b_weights) != base["expected_modules"] or any(torch.count_nonzero(p) for p in b_weights):
        raise ValueError("Unexpected target count or nonzero initial LoRA B")
    if any(p.dtype != torch.float32 or ".lora_" not in n for n, p in params):
        raise ValueError("Expected FP32 adapters and a frozen backbone")
    return model


def train(model_key, cfg, tokenizer, examples, out):
    tuning, base = cfg["fine_tuning"], cfg["bases"][model_key]
    targets = [sum(t != -100 for t in labels[1:]) for _, labels in examples]
    observed_counts = dict(input_tokens=sum(len(ids) for ids, _ in examples),
                           supervised_tokens=sum(targets),
                           max_input_length=max(len(ids) for ids, _ in examples))
    if len(examples) != cfg["training_data"]["train"]["rows"] or observed_counts != cfg["training_data"]["expected_counts_per_epoch"][model_key]:
        raise ValueError(f"Training examples disagree with verified protocol: {len(examples)}, {observed_counts}")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rng = np.random.default_rng(tuning["torch_seed"])
    orders = [rng.permutation(len(examples)).tolist() for _ in range(tuning["epochs"])]
    write_json(out / "row_order.json", orders)
    torch.manual_seed(tuning["torch_seed"])
    started = time.perf_counter()
    net = new_adapter(load_base(base, torch.bfloat16, training=True), base, tuning)
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tuning["lr"], betas=tuple(tuning["betas"]),
                                  eps=tuning["epsilon"], weight_decay=tuning["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, 12, tuning["optimizer_updates"])
    mb = tuning["microbatch"]
    group_size = mb * tuning["gradient_accumulation"]
    step = 0
    torch.cuda.reset_peak_memory_stats()
    epochs = []
    with (out / "training_trace.jsonl").open("w", buffering=1) as trace:
        for epoch, order in enumerate(orders, start=1):
            epoch_nll, epoch_tokens = 0., 0
            for first in range(0, len(order), group_size):
                group = order[first:first + group_size]
                count = sum(targets[idx] for idx in group)
                optimizer.zero_grad(set_to_none=True)
                nll_sum = torch.zeros((), device=DEVICE)
                for offset in range(0, len(group), mb):
                    micro = [examples[idx] for idx in group[offset:offset + mb]]
                    batch = {k: v.to(DEVICE) for k, v in collate_rows(micro, pad).items()}
                    labels = batch.pop("labels")[:, 1:]
                    active = labels != -100
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(**batch, use_cache=False).logits[:, :-1]
                        nll = F.cross_entropy(logits[active].float(), labels[active], reduction="sum")
                    (nll / count).backward()
                    nll_sum += nll.detach()
                    del logits, nll
                grad_norm = torch.nn.utils.clip_grad_norm_(params, tuning["clip_gradient_norm"],
                                                         error_if_nonfinite=True)
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step += 1
                value = float(nll_sum.item())
                epoch_nll += value
                epoch_tokens += count
                row = dict(step=step, epoch=epoch, examples=len(group), supervised_tokens=count,
                           answer_nll=value/count, lr=lr, gradient_norm=float(grad_norm))
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 1 or step % 10 == 0:
                    print(f"{model_key} optimizer step {step}/{tuning['optimizer_updates']} loss {value/count:.4f}", flush=True)
            epochs.append(dict(epoch=epoch, examples=len(order), supervised_tokens=epoch_tokens,
                               online_answer_nll=epoch_nll/epoch_tokens))
    if step != tuning["optimizer_updates"]:
        raise ValueError(f"Unexpected optimizer step count {step}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    adapter = out / "adapter"
    net.peft_config["default"].base_model_name_or_path = base["model"]
    net.peft_config["default"].revision = base["revision"]
    with TemporaryDirectory(dir=out) as scratch:
        target = Path(scratch) / "adapter"
        net.save_pretrained(target, safe_serialization=True, save_embedding_layers=False)
        target.rename(adapter)
    info = dict(model=model_key, base=base, training_dataset=cfg["training_data"]["dataset"],
                training_dataset_revision=cfg["training_data"]["revision"], training_split="train",
                row_order_file="row_order.json", counts_per_epoch=observed_counts,
                trained_steps=step, epochs=epochs, adapter_parameters=sum(p.numel() for p in params),
                training_wall_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(DEVICE),
                packages={name: version(name) for name in ("torch","transformers","peft","accelerate","numpy","pyarrow")})
    write_json(out / "training.json", info)


@torch.no_grad()
def check_zero_export(model_key, cfg, tokenizer, out):
    """Perform the same FP32 merge/FP16 cast with zero LoRA and reproduce cached base scores."""
    base = cfg["bases"][model_key]
    model = new_adapter(load_base(base, torch.float32), base, cfg["fine_tuning"])
    merged = model.merge_and_unload(safe_merge=True).half().eval()
    # Reload the serialized FP16 weights as the public scorer does. Nonpersistent
    # rotary buffers in GPT-NeoX are recomputed by from_pretrained at load time.
    with TemporaryDirectory(dir=out) as scratch:
        merged.save_pretrained(scratch, safe_serialization=True)
        del model, merged
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            scratch, torch_dtype=torch.float16, device_map={"": DEVICE}, local_files_only=True
        ).eval()
    with (ROOT / cfg["witness_audit"]["inputs"] / "train.jsonl").open() as f:
        rows = [json.loads(next(f)) for _ in range(4)]
    with (ROOT / BASE_SCORES[model_key] / "train_scores.jsonl").open() as f:
        cached = [json.loads(next(f)) for _ in rows]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [r["text"] for r in rows]
    token_scores = extract_token_logprobs(texts, model, tokenizer, DEVICE, max_length=512)
    checks = []
    for row, original, lp in zip(rows, cached, token_scores):
        if row["id"] != original["id"] or row["label"] != original["label"]:
            raise ValueError("Cached base-score records differ from fixed construction batch")
        computed = compute_all_scores(row["text"], lp, (20,))
        keys = ("mean_logprob", "min_k_20_logprob")
        checks.append(dict(id=row["id"], zero_delta={k: round(computed[k], 6) for k in keys},
                           base={k: original[k] for k in keys}))
    write_json(out / "zero_delta.json", checks)
    if any(row["zero_delta"] != row["base"] for row in checks):
        raise ValueError("Zero-delta FP16 checkpoint differs from cached FP16 base scores")


@torch.no_grad()
def export_final(model_key, cfg, tokenizer, out):
    started = time.perf_counter()
    base = cfg["bases"][model_key]
    check_zero_export(model_key, cfg, tokenizer, out)
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(base, torch.float32), str(out / "adapter"))
    merged = model.merge_and_unload(safe_merge=True).half().eval()
    with TemporaryDirectory(dir=out) as scratch:
        target = Path(scratch) / "merged"
        merged.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        target.rename(out / "merged")
    write_json(out / "export.json", dict(model=model_key, base=base, merge_dtype="float32",
                                          export_dtype="float16", export_wall_seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pythia", "biogpt"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Optional directory containing the pinned downstream release")
    parser.add_argument("--protocol", type=Path, default=DESIGN)
    parser.add_argument("--export-only", action="store_true", help="Export a completed adapter without retraining")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.protocol.read_text())
    base = cfg["bases"][args.model]
    model_id = base["model"].replace("microsoft/BioGPT-Large", "microsoft/biogpt-large")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=base["revision"], local_files_only=True)
    if type(tokenizer).__name__ != base["tokenizer_class"] or tokenizer.padding_side != "right":
        raise ValueError("Expected pinned native tokenizer with right padding")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "merged").exists() or (not args.export_only and (out / "training_trace.jsonl").exists()):
        raise FileExistsError("Output contains a previous run; choose a fresh directory or use --export-only")
    if not args.export_only:
        data = lamini_data
        data_dir = args.data_dir or ROOT / "artifacts/lamini_design"
        if cfg["training_data"]["dataset"] == "databricks/databricks-dolly-15k":
            import dolly_data as data
            data_dir = args.data_dir or ROOT / "artifacts/dolly_matched/data"
            preflight = json.loads((data_dir / f"{args.model}_preflight.json").read_text())
            counts = {key: preflight[key] for key in ("input_tokens", "supervised_tokens", "max_input_length")}
            cfg["training_data"]["expected_counts_per_epoch"] = {args.model: counts}
        rows = data.load_split(data_dir, "train")
        examples = [data.encode_row(r, tokenizer) for r in rows]
        write_json(out / "effective_train_config.json", cfg)
        train(args.model, cfg, tokenizer, examples, out)
        gc.collect()
        torch.cuda.empty_cache()
    export_final(args.model, cfg, tokenizer, out)
    print(f"Completed {args.model}: {out}", flush=True)


if __name__ == "__main__":
    main()
