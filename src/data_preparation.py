"""Deterministic HH-RLHF data preparation for the final SFT-vs-DPO experiments.

The functions in this module are intentionally shared by both final notebooks so
that SFT and DPO train and evaluate on exactly the same underlying preference
pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, load_dataset


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
DATASET_NAME = "Anthropic/hh-rlhf"
SEED = 42
TRAIN_SIZE = 5_000
TEST_SIZE = 1_000
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
ASSISTANT_SEPARATOR = "\n\nAssistant:"


@dataclass(frozen=True)
class SplitInfo:
    dataset_name: str
    seed: int
    train_size: int
    test_size: int
    max_length: int
    max_prompt_length: int
    train_valid_before_sampling: int
    test_valid_before_sampling: int
    train_selected: int
    test_selected: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "train_valid_before_sampling": self.train_valid_before_sampling,
            "test_valid_before_sampling": self.test_valid_before_sampling,
            "train_selected": self.train_selected,
            "test_selected": self.test_selected,
        }


def parse_hh_rlhf_pair(example: Dict[str, str], index: Optional[int] = None) -> Dict[str, Any]:
    """Parse one HH-RLHF row into a shared prompt and two final responses.

    HH-RLHF stores complete conversations in both ``chosen`` and ``rejected``.
    This parser splits both strings at the final assistant marker and keeps the
    pair only if the resulting prompts are identical and both responses are
    non-empty.
    """

    chosen_text = example.get("chosen", "")
    rejected_text = example.get("rejected", "")

    chosen_parts = chosen_text.rsplit(ASSISTANT_SEPARATOR, 1)
    rejected_parts = rejected_text.rsplit(ASSISTANT_SEPARATOR, 1)

    if len(chosen_parts) != 2 or len(rejected_parts) != 2:
        prompt = chosen_response = rejected_response = ""
        is_parseable = False
    else:
        chosen_prompt = chosen_parts[0] + ASSISTANT_SEPARATOR
        rejected_prompt = rejected_parts[0] + ASSISTANT_SEPARATOR
        chosen_response = chosen_parts[1].strip()
        rejected_response = rejected_parts[1].strip()
        same_prompt = chosen_prompt == rejected_prompt
        nonempty = bool(chosen_response) and bool(rejected_response)
        is_parseable = same_prompt and nonempty
        prompt = chosen_prompt if is_parseable else ""
        if not is_parseable:
            chosen_response = ""
            rejected_response = ""

    parsed = {
        "source_index": index if index is not None else -1,
        "prompt": prompt,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
        "is_parseable": is_parseable,
    }
    return parsed


def add_token_lengths(example: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Add token lengths used for validity filtering."""

    prompt_text = example["prompt"] + " "
    chosen_text = prompt_text + example["chosen_response"]
    rejected_text = prompt_text + example["rejected_response"]

    prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
    chosen_tokens = len(tokenizer(chosen_text, add_special_tokens=True)["input_ids"])
    rejected_tokens = len(tokenizer(rejected_text, add_special_tokens=True)["input_ids"])

    return {
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
    }


def is_valid_pair(example: Dict[str, Any], max_length: int = MAX_LENGTH, max_prompt_length: int = MAX_PROMPT_LENGTH) -> bool:
    """Return True for parseable, non-empty pairs that fit the configured limits."""

    return (
        bool(example["is_parseable"])
        and 0 < len(example["prompt"])
        and 0 < len(example["chosen_response"])
        and 0 < len(example["rejected_response"])
        and example["prompt_tokens"] <= max_prompt_length
        and example["chosen_tokens"] <= max_length
        and example["rejected_tokens"] <= max_length
    )


def _parse_filter_and_sample_split(
    split: Dataset,
    tokenizer: Any,
    sample_size: int,
    seed: int,
    max_length: int,
    max_prompt_length: int,
) -> Tuple[Dataset, int]:
    parsed = split.map(
        parse_hh_rlhf_pair,
        with_indices=True,
        remove_columns=split.column_names,
        desc="Parsing HH-RLHF preference pairs",
    )
    with_lengths = parsed.map(
        lambda ex: add_token_lengths(ex, tokenizer),
        desc="Computing token lengths",
    )
    valid = with_lengths.filter(
        lambda ex: is_valid_pair(ex, max_length=max_length, max_prompt_length=max_prompt_length),
        desc="Filtering valid preference pairs",
    )

    if len(valid) < sample_size:
        raise ValueError(
            f"Requested {sample_size} valid pairs, but only {len(valid)} are available "
            f"after parsing and filtering."
        )

    sampled = valid.shuffle(seed=seed).select(range(sample_size))
    return sampled, len(valid)


def prepare_hh_rlhf_splits(
    tokenizer: Any,
    dataset_name: str = DATASET_NAME,
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
    max_length: int = MAX_LENGTH,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
) -> Tuple[Dataset, Dataset, SplitInfo]:
    """Load, parse, filter, and deterministically sample final train/test pairs.

    Parsing and validity filtering are applied to the full official split before
    final sampling. Calling this function with the same arguments yields the same
    selected ``source_index`` values in both notebooks.
    """

    raw = load_dataset(dataset_name)
    train_pairs, train_valid = _parse_filter_and_sample_split(
        raw["train"], tokenizer, train_size, seed, max_length, max_prompt_length
    )
    test_pairs, test_valid = _parse_filter_and_sample_split(
        raw["test"], tokenizer, test_size, seed, max_length, max_prompt_length
    )

    info = SplitInfo(
        dataset_name=dataset_name,
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        train_valid_before_sampling=train_valid,
        test_valid_before_sampling=test_valid,
        train_selected=len(train_pairs),
        test_selected=len(test_pairs),
    )
    return train_pairs, test_pairs, info


def to_sft_text_dataset(pairs: Dataset) -> Dataset:
    """Create SFT text examples from the chosen responses only."""

    def convert(example: Dict[str, Any]) -> Dict[str, str]:
        return {"text": example["prompt"] + " " + example["chosen_response"]}

    return pairs.map(convert, remove_columns=pairs.column_names, desc="Creating SFT texts")


def tokenize_sft_example(example: Dict[str, str], tokenizer: Any, max_length: int = MAX_LENGTH) -> Dict[str, Any]:
    """Tokenize one SFT example and mask prompt tokens in the labels."""

    text = example["text"]
    marker = ASSISTANT_SEPARATOR + " "
    response_start = text.rfind(marker)
    if response_start == -1:
        raise ValueError("SFT example does not contain the expected assistant marker.")
    response_start += len(marker)

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = list(encoding["input_ids"])
    for i, (start, end) in enumerate(encoding["offset_mapping"]):
        if end <= response_start:
            labels[i] = -100

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def to_dpo_dataset(pairs: Dataset) -> Dataset:
    """Create the prompt/chosen/rejected format expected by TRL DPOTrainer."""

    def convert(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": example["prompt"] + " ",
            "chosen": example["chosen_response"],
            "rejected": example["rejected_response"],
        }

    return pairs.map(convert, remove_columns=pairs.column_names, desc="Creating DPO records")
