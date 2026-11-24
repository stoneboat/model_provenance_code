"""Prepare the complete Dolly release for the one-epoch matched downstream audit."""
from functools import lru_cache
import json
from pathlib import Path

FILENAME = "databricks-dolly-15k.jsonl"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
NUM_ROWS = 15011
MAX_LENGTH = 512


def load_split(data_dir, split):
    if split != "train":
        raise ValueError("Dolly-15K publishes only a training split")
    with (Path(data_dir) / FILENAME).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != NUM_ROWS:
        raise ValueError(f"Expected {NUM_ROWS} Dolly rows, found {len(rows)}")
    return rows


def question_for(row):
    question = row["instruction"]
    return question + "\n" + row["context"] if row["context"] else question


@lru_cache(maxsize=4)
def _cues(tokenizer):
    return (tokenizer.encode("Question:", add_special_tokens=False),
            tokenizer.encode("\nAnswer:", add_special_tokens=False))


def encode_with_counts(row, tokenizer):
    question = question_for(row)
    raw_prompt = tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
    answer = tokenizer.encode(" " + row["response"], add_special_tokens=False)
    if not row["response"].strip() or not answer:
        raise ValueError("Dolly record has an empty answer")
    native_prompt = tokenizer.build_inputs_with_special_tokens(raw_prompt)
    full_length = len(native_prompt) + len(answer) + 1
    original_answer = len(answer)
    instruction_cut = context_cut = False
    if full_length > MAX_LENGTH:
        # Allocate to the instruction and the response before optional context.
        # Very long instructions still leave room for real response text and EOS.
        prefix, suffix = _cues(tokenizer)
        overhead = len(tokenizer.build_inputs_with_special_tokens(prefix + suffix)) + 1
        available = MAX_LENGTH - overhead
        instruction = tokenizer.encode(" " + row["instruction"], add_special_tokens=False)
        context = (tokenizer.encode("\n" + row["context"], add_special_tokens=False)
                   if row["context"] else [])
        min_response = 1
        while not tokenizer.decode(answer[:min_response]).strip():
            min_response += 1
            if min_response > len(answer):
                raise ValueError("Dolly response has no decodable text")
        keep_instruction = min(len(instruction), available - min_response)
        if not tokenizer.decode(instruction[:keep_instruction]).strip():
            raise ValueError("No instruction text retained")
        instruction_cut = keep_instruction < len(instruction)
        answer = answer[:available - keep_instruction]
        keep_context = available - keep_instruction - len(answer)
        context_cut = keep_context < len(context)
        raw_prompt = prefix + instruction[:keep_instruction] + context[:keep_context] + suffix
        native_prompt = tokenizer.build_inputs_with_special_tokens(raw_prompt)
    sequence = tokenizer.build_inputs_with_special_tokens(raw_prompt + answer)
    if sequence != native_prompt + answer:
        raise ValueError("Native tokenizer changed supervised answer suffix")
    targets = answer + [tokenizer.eos_token_id]
    ids = sequence + [tokenizer.eos_token_id]
    if len(ids) > MAX_LENGTH or len(targets) < 2:
        raise ValueError("Invalid cropped Dolly record")
    labels = [-100] * len(native_prompt) + targets
    return ids, labels, (full_length, original_answer, len(answer), instruction_cut, context_cut)


def encode_row(row, tokenizer):
    ids, labels, _ = encode_with_counts(row, tokenizer)
    return ids, labels
