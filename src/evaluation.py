"""Shared preference-discrimination evaluation for Base, SFT, and DPO."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@torch.no_grad()
def response_mean_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: Optional[int] = None,
) -> float:
    """Mean log-probability of response tokens conditioned on the prompt."""

    prompt_text = prompt + " " if not prompt.endswith(" ") else prompt
    full_text = prompt_text + response

    full = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = full.pop("offset_mapping")[0].tolist()
    input_ids = full["input_ids"]
    attention_mask = full.get("attention_mask")

    if max_length is not None and input_ids.shape[1] > max_length:
        raise ValueError(f"Sequence has {input_ids.shape[1]} tokens, exceeding max_length={max_length}.")

    response_start = len(prompt_text)
    response_label_positions = [
        label_position
        for label_position, (_start, end) in enumerate(offsets[1:])
        if end > response_start
    ]
    if not response_label_positions:
        raise ValueError("No response tokens were available to score.")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_log_probs = token_log_probs[:, response_label_positions]
    return float(response_log_probs.mean().item())


def summarize_preference_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    margins = [float(row["margin"]) for row in rows]
    correct = [bool(row["correct"]) for row in rows]
    correct_count = sum(correct)
    evaluated = len(rows)
    return {
        "evaluated": evaluated,
        "correct": correct_count,
        "accuracy": correct_count / evaluated if evaluated else float("nan"),
        "mean_margin": mean(margins) if margins else float("nan"),
        "median_margin": median(margins) if margins else float("nan"),
        "min_margin": min(margins) if margins else float("nan"),
        "max_margin": max(margins) if margins else float("nan"),
    }


def save_results_csv(rows: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "source_index",
        "chosen_score",
        "rejected_score",
        "margin",
        "correct",
        "prompt_tokens",
        "chosen_tokens",
        "rejected_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def evaluate_preference_pairs(
    model: Any,
    tokenizer: Any,
    pairs: Iterable[Dict[str, Any]],
    output_csv: Optional[Union[str, Path]] = None,
    model_label: str = "model",
    max_length: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate preference accuracy from chosen-vs-rejected mean log-probs.

    A positive margin, ``chosen_score - rejected_score > 0``, counts as a
    correct preference prediction.
    """

    model.eval()
    rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(tqdm(pairs, desc=f"Evaluating {model_label}")):
        chosen_score = response_mean_logprob(
            model, tokenizer, example["prompt"], example["chosen_response"], max_length=max_length
        )
        rejected_score = response_mean_logprob(
            model, tokenizer, example["prompt"], example["rejected_response"], max_length=max_length
        )
        margin = chosen_score - rejected_score
        rows.append(
            {
                "example_id": idx,
                "source_index": example.get("source_index", ""),
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": margin,
                "correct": margin > 0,
                "prompt_tokens": example.get("prompt_tokens", ""),
                "chosen_tokens": example.get("chosen_tokens", ""),
                "rejected_tokens": example.get("rejected_tokens", ""),
            }
        )

    stats = summarize_preference_results(rows)
    if output_csv is not None:
        save_results_csv(rows, output_csv)
    return rows, stats


def print_preference_summary(label: str, stats: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"{label} — HELD-OUT PREFERENCE EVALUATION")
    print("=" * 60)
    print(f"Evaluated pairs: {stats['evaluated']}")
    print(f"Preference accuracy: {stats['correct']}/{stats['evaluated']} ({stats['accuracy'] * 100:.2f}%)")
    print("Preference margin (chosen - rejected):")
    print(f"  Mean:   {stats['mean_margin']:.4f}")
    print(f"  Median: {stats['median_margin']:.4f}")
    print(f"  Min:    {stats['min_margin']:.4f}")
    print(f"  Max:    {stats['max_margin']:.4f}")
