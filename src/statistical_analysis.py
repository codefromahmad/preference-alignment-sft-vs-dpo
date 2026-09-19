"""Statistical analysis for the final Base-vs-SFT-vs-DPO experiment.

The script expects per-example evaluation CSVs produced by the shared evaluator:

- results/base_preference_eval.csv
- results/sft_preference_eval.csv
- results/dpo_preference_eval.csv

It verifies that all three files contain the same 1,000 source_index values in
the same order, then computes paired accuracy and margin analyses.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_N = 1_000

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
INPUT_FILES = {
    "Base": RESULTS_DIR / "base_preference_eval.csv",
    "SFT": RESULTS_DIR / "sft_preference_eval.csv",
    "DPO": RESULTS_DIR / "dpo_preference_eval.csv",
}
OUTPUT_JSON = RESULTS_DIR / "statistical_analysis.json"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def read_results(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")

    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"source_index", "chosen_score", "rejected_score", "margin", "correct"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "source_index": str(row["source_index"]),
                        "chosen_score": float(row["chosen_score"]),
                        "rejected_score": float(row["rejected_score"]),
                        "margin": float(row["margin"]),
                        "correct": parse_bool(row["correct"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - include CSV row context.
                raise ValueError(f"Failed to parse {path} row {row_number}: {exc}") from exc

    return rows


def verify_same_examples(results: Dict[str, List[Dict[str, object]]]) -> List[str]:
    source_indices = {model: [str(row["source_index"]) for row in rows] for model, rows in results.items()}

    for model, values in source_indices.items():
        if len(values) != EXPECTED_N:
            raise ValueError(f"{model} contains {len(values)} rows; expected exactly {EXPECTED_N}.")
        if len(set(values)) != len(values):
            raise ValueError(f"{model} contains duplicate source_index values.")

    reference_name = "Base"
    reference = source_indices[reference_name]
    for model, values in source_indices.items():
        if values != reference:
            first_mismatch = next((i for i, (a, b) in enumerate(zip(reference, values)) if a != b), None)
            if first_mismatch is None and len(values) != len(reference):
                first_mismatch = min(len(values), len(reference))
            raise ValueError(
                f"source_index order mismatch between {reference_name} and {model} "
                f"at position {first_mismatch}: {reference_name}={reference[first_mismatch] if first_mismatch is not None and first_mismatch < len(reference) else 'NA'}, "
                f"{model}={values[first_mismatch] if first_mismatch is not None and first_mismatch < len(values) else 'NA'}"
            )

    return reference


def model_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    correct = [bool(row["correct"]) for row in rows]
    margins = [float(row["margin"]) for row in rows]
    correct_count = sum(correct)
    n = len(rows)
    return {
        "evaluated_pairs": n,
        "correct": correct_count,
        "preference_accuracy": correct_count / n,
        "preference_accuracy_percent": 100.0 * correct_count / n,
        "mean_preference_margin": mean(margins),
        "median_preference_margin": median(margins),
    }


def exact_two_sided_binomial_pvalue(successes: int, trials: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for probability p.

    For McNemar's exact test with p=0.5, successes is the smaller discordant
    count and trials is the total number of discordant pairs.
    """

    if trials == 0:
        return 1.0
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if p != 0.5:
        raise NotImplementedError("Only p=0.5 is implemented for this analysis.")

    lower_tail = 0.0
    probability = 2.0 ** (-trials)
    for k in range(0, successes + 1):
        if k == 0:
            probability = 2.0 ** (-trials)
        elif k > 0:
            probability *= (trials - k + 1) / k
        lower_tail += probability
    return min(1.0, 2.0 * lower_tail)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_ci(values: Sequence[float], seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty vector.")

    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(resamples):
        total = 0.0
        for _sample_idx in range(n):
            total += values[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()
    return {
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def wilcoxon_signed_rank(differences: Sequence[float]) -> Dict[str, float]:
    nonzero = [diff for diff in differences if diff != 0.0]
    n = len(nonzero)
    if n == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "method": "all differences are zero",
        }

    abs_diffs = [abs(diff) for diff in nonzero]
    ranks = average_ranks(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    total_rank = sum(ranks)
    w_minus = total_rank - w_plus
    expected = total_rank / 2.0
    variance = sum(rank * rank for rank in ranks) / 4.0
    if variance == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = (w_plus - expected) / math.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))

    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": min(w_plus, w_minus),
        "z": z,
        "p_value": p_value,
        "method": "normal approximation with tie-aware average ranks",
    }


def paired_comparison(name_a: str, rows_a: Sequence[Dict[str, object]], name_b: str, rows_b: Sequence[Dict[str, object]]) -> Dict[str, object]:
    correct_a = [bool(row["correct"]) for row in rows_a]
    correct_b = [bool(row["correct"]) for row in rows_b]
    margins_a = [float(row["margin"]) for row in rows_a]
    margins_b = [float(row["margin"]) for row in rows_b]

    accuracy_diffs = [float(b) - float(a) for a, b in zip(correct_a, correct_b)]
    margin_diffs = [b - a for a, b in zip(margins_a, margins_b)]

    a_wrong_b_correct = sum((not a) and b for a, b in zip(correct_a, correct_b))
    a_correct_b_wrong = sum(a and (not b) for a, b in zip(correct_a, correct_b))
    discordant = a_wrong_b_correct + a_correct_b_wrong
    mcnemar_p = exact_two_sided_binomial_pvalue(min(a_wrong_b_correct, a_correct_b_wrong), discordant)

    accuracy_diff = mean(accuracy_diffs)
    margin_diff = mean(margin_diffs)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "direction": f"{name_b} minus {name_a}",
        "paired_accuracy_difference": accuracy_diff,
        "paired_accuracy_difference_percentage_points": 100.0 * accuracy_diff,
        "a_wrong_b_correct": a_wrong_b_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p_value": mcnemar_p,
        "accuracy_difference_bootstrap_95_ci": paired_bootstrap_ci(accuracy_diffs),
        "mean_paired_preference_margin_difference": margin_diff,
        "margin_difference_bootstrap_95_ci": paired_bootstrap_ci(margin_diffs),
        "wilcoxon_signed_rank_margin_difference": wilcoxon_signed_rank(margin_diffs),
    }


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_pp(value: float) -> str:
    return f"{value:.2f} pp"


def print_summary(summary: Dict[str, object]) -> None:
    print("Final SFT-vs-DPO statistical analysis")
    print("=" * 48)
    print(f"Verified paired source_index order: {summary['paired_source_index_count']} examples")
    print()

    print("Model summaries")
    print("---------------")
    for model, stats in summary["models"].items():
        print(
            f"{model}: n={stats['evaluated_pairs']}, "
            f"correct={stats['correct']}, "
            f"accuracy={format_percent(stats['preference_accuracy'])}, "
            f"mean margin={stats['mean_preference_margin']:.4f}, "
            f"median margin={stats['median_preference_margin']:.4f}"
        )
    print()

    print("Paired comparisons (B minus A)")
    print("------------------------------")
    for name, comparison in summary["paired_comparisons"].items():
        acc_ci = comparison["accuracy_difference_bootstrap_95_ci"]
        margin_ci = comparison["margin_difference_bootstrap_95_ci"]
        wilcoxon = comparison["wilcoxon_signed_rank_margin_difference"]
        print(f"{name}:")
        print(
            f"  accuracy diff = {format_pp(comparison['paired_accuracy_difference_percentage_points'])} "
            f"(95% bootstrap CI {format_pp(100.0 * acc_ci['lower'])}, {format_pp(100.0 * acc_ci['upper'])})"
        )
        print(
            f"  discordant counts: A wrong/B correct={comparison['a_wrong_b_correct']}, "
            f"A correct/B wrong={comparison['a_correct_b_wrong']}; "
            f"McNemar exact p={comparison['mcnemar_exact_two_sided_p_value']:.6g}"
        )
        print(
            f"  mean margin diff = {comparison['mean_paired_preference_margin_difference']:.4f} "
            f"(95% bootstrap CI {margin_ci['lower']:.4f}, {margin_ci['upper']:.4f})"
        )
        print(
            f"  Wilcoxon signed-rank p={wilcoxon['p_value']:.6g} "
            f"(n_nonzero={wilcoxon['n_nonzero']}, method={wilcoxon['method']})"
        )
    print()
    print(f"Saved JSON summary: {OUTPUT_JSON}")


def main() -> None:
    results = {model: read_results(path) for model, path in INPUT_FILES.items()}
    source_indices = verify_same_examples(results)

    summary: Dict[str, object] = {
        "input_files": {model: str(path.relative_to(REPO_ROOT)) for model, path in INPUT_FILES.items()},
        "paired_source_index_count": len(source_indices),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "models": {model: model_summary(rows) for model, rows in results.items()},
        "paired_comparisons": {},
    }

    comparisons = [("Base", "SFT"), ("Base", "DPO"), ("SFT", "DPO")]
    for name_a, name_b in comparisons:
        key = f"{name_a}_vs_{name_b}"
        summary["paired_comparisons"][key] = paired_comparison(name_a, results[name_a], name_b, results[name_b])

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
