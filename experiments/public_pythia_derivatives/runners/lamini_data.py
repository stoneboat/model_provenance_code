"""Data format shared by the matched Lamini fine-tuning and loss evaluations."""
from pathlib import Path

import pyarrow.parquet as pq
import torch


SPLIT_FILES = {
    "train": ("train-00000-of-00001-5cdebbc48da41394.parquet", 1260),
    "test": ("test-00000-of-00001-4c77a066a883f339.parquet", 140),
}


def load_split(data_dir, split):
    filename, expected = SPLIT_FILES[split]
    rows = pq.read_table(Path(data_dir) / filename, columns=["question", "answer"]).to_pylist()
    if len(rows) != expected:
        raise ValueError(f"Lamini {split} has {len(rows)} rows, expected {expected}")
    return rows


def encode_row(row, tokenizer):
    prefix = f"Question: {row['question']}\nAnswer:"
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = (tokenizer.encode(" " + row["answer"], add_special_tokens=False)
                  if row["answer"].strip() else [])
    # The native tokenizers retain the suffix; BioGPT prepends a context token.
    native_prompt = tokenizer.build_inputs_with_special_tokens(prompt_ids)
    sequence = tokenizer.build_inputs_with_special_tokens(prompt_ids + answer_ids)
    if sequence != native_prompt + answer_ids:
        raise ValueError("The native tokenizer changed the answer-token suffix")
    input_ids = sequence + [tokenizer.eos_token_id]
    if len(input_ids) > 512:
        raise ValueError(f"Unexpected truncation in pinned Lamini data: {len(input_ids)} tokens")
    labels = [-100] * len(native_prompt) + answer_ids + [tokenizer.eos_token_id]
    return input_ids, labels


def collate_rows(examples, pad_token_id):
    length = 8 * ((max(len(ids) for ids, _ in examples) + 7) // 8)
    input_ids = []
    masks = []
    labels = []
    for ids, targets in examples:
        padding = length - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        masks.append([1] * len(ids) + [0] * padding)
        labels.append(targets + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
