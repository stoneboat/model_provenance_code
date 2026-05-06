"""Dataset loading utilities for MIMIR GitHub shard-membership auditing.

MIMIR dataset schema (iamgroot42/mimir):
  - Configs (subsets): arxiv, dm_mathematics, github, hackernews, pile_cc,
      pubmed_central, wikipedia_(en), full_pile, c4, temporal_arxiv, temporal_wiki
  - N-gram splits: ngram_7_0.2, ngram_13_0.2, ngram_13_0.8 (for most sources)
  - Columns: member (str), nonmember (str),
             member_neighbors (List[str]), nonmember_neighbors (List[str])

IMPORTANT — naming convention in the raw files:
  cache_100_200_1000_512/train/<config>_<split>.jsonl  → MEMBER texts
  cache_100_200_1000_512/test/<config>_<split>.jsonl   → NONMEMBER texts

  "train" refers to examples IN the model's training corpus (members).
  "test"  refers to examples NOT in the model's training corpus (nonmembers).
  These are NOT train/test splits for MIA evaluation — that split is created
  downstream in splitting.py from the member and nonmember pools.

Default choices for the experiment:
  config  = "github"
  split   = "ngram_13_0.2"   (13-gram strict deduplication, standard in MIMIR papers)
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DATASET_ID = "iamgroot42/mimir"
KNOWN_CONFIGS = [
    "arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
    "pubmed_central", "wikipedia_(en)", "full_pile", "c4",
    "temporal_arxiv", "temporal_wiki",
]
KNOWN_SPLITS = ["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8", "none"]
CACHE_PREFIX = "cache_100_200_1000_512"
CACHE_PREFIX_FULL_PILE = "cache_100_200_10000_512"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_prefix(config: str) -> str:
    return CACHE_PREFIX_FULL_PILE if config == "full_pile" else CACHE_PREFIX


def _member_filename(config: str, split: str) -> str:
    suffix = f"_{split}" if split != "none" else ""
    return f"{_cache_prefix(config)}/train/{config}{suffix}.jsonl"


def _nonmember_filename(config: str, split: str) -> str:
    suffix = f"_{split}" if split != "none" else ""
    return f"{_cache_prefix(config)}/test/{config}{suffix}.jsonl"


def list_configs(token: Optional[str] = None) -> list:
    """Return available configs for the MIMIR dataset."""
    try:
        from datasets import get_dataset_config_names  # noqa: PLC0415
        return get_dataset_config_names(DATASET_ID, token=token)
    except Exception as e:
        logger.warning("Could not list configs via datasets API (%s); returning known list.", e)
        return KNOWN_CONFIGS


def list_splits(config: str = "github", token: Optional[str] = None) -> list:
    """Return available n-gram splits for a given config."""
    try:
        from datasets import get_dataset_split_names  # noqa: PLC0415
        return get_dataset_split_names(DATASET_ID, config_name=config, token=token)
    except Exception as e:
        logger.warning("Could not list splits via datasets API (%s); returning known list.", e)
        return KNOWN_SPLITS


def load_texts_from_jsonl(local_path: str) -> list:
    """Load one text per line from a MIMIR JSONL file (each line is a JSON string)."""
    import json  # noqa: PLC0415
    texts = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, dict) and "member" in obj:
                texts.append(obj["member"])
            # Skip anything else silently
    return texts


def load_mimir_github_via_hub_download(
    config: str = "github",
    split: str = "ngram_13_0.2",
    token: Optional[str] = None,
) -> tuple:
    """Download MIMIR JSONL files directly and return (members, nonmembers).

    File mapping (from mimir.py dataset script):
      members:    cache_100_200_1000_512/train/<config>_<split>.jsonl
      nonmembers: cache_100_200_1000_512/test/<config>_<split>.jsonl

    Returns:
        (member_texts: list[str], nonmember_texts: list[str])
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    member_file = _member_filename(config, split)
    nonmember_file = _nonmember_filename(config, split)

    logger.info("Downloading member file:    %s", member_file)
    member_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=member_file,
        repo_type="dataset",
        token=token,
    )
    logger.info("Downloading nonmember file: %s", nonmember_file)
    nonmember_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=nonmember_file,
        repo_type="dataset",
        token=token,
    )

    members = load_texts_from_jsonl(member_path)
    nonmembers = load_texts_from_jsonl(nonmember_path)

    logger.info(
        "Loaded %d members from %s", len(members), member_file
    )
    logger.info(
        "Loaded %d nonmembers from %s", len(nonmembers), nonmember_file
    )
    return members, nonmembers


def load_mimir_github(
    config: str = "github",
    split: str = "ngram_13_0.2",
    token: Optional[str] = None,
) -> tuple:
    """Load MIMIR GitHub member/nonmember texts via direct hub download.

    Returns:
        (member_texts: list[str], nonmember_texts: list[str])
    """
    return load_mimir_github_via_hub_download(config, split, token)


def load_mimir_github_nonmembers(
    config: str = "github",
    split: str = "ngram_13_0.2",
    token: Optional[str] = None,
) -> list:
    """Download only the MIMIR GitHub nonmember (test-side) texts.

    Returns:
        nonmember_texts: list[str]
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    nonmember_file = _nonmember_filename(config, split)
    logger.info("Downloading nonmember file: %s", nonmember_file)
    nonmember_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=nonmember_file,
        repo_type="dataset",
        token=token,
    )
    nonmembers = load_texts_from_jsonl(nonmember_path)
    logger.info("Loaded %d nonmembers from %s", len(nonmembers), nonmember_file)
    return nonmembers


def texts_to_records(
    texts: list,
    label: int,
    source: str = "mimir_github",
    split_origin: str = "",
    id_prefix: str = "",
) -> list:
    """Convert raw texts to normalized record dicts.

    Each record:
        id, text, label, source, split_origin, text_hash
    """
    records = []
    for i, text in enumerate(texts):
        rec = {
            "id": f"{id_prefix}{i:06d}",
            "text": text,
            "label": label,
            "source": source,
            "split_origin": split_origin,
            "text_hash": _text_hash(text),
        }
        records.append(rec)
    return records
