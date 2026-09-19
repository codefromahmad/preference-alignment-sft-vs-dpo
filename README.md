# Evaluating Preference Alignment Methods for Compact Language Models:
# A Comparison of Supervised Fine-Tuning and Direct Preference Optimization

## Overview

This repository compares three conditions for preference alignment of a compact language model under a limited-data, resource-constrained setup:

- the pretrained Base model,
- Supervised Fine-Tuning (SFT), and
- Direct Preference Optimization (DPO).

All experiments use `HuggingFaceTB/SmolLM2-360M` and Anthropic HH-RLHF preference data. The research question is:

> How do Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) compare in aligning a compact language model with human preferences under limited training data?

The results are specific to this compact-model, limited-data, one-epoch configuration and should not be interpreted as showing universal superiority of either SFT or DPO.

## Experimental Design

- **Model:** `HuggingFaceTB/SmolLM2-360M`
- **Parameters:** 361,821,120
- **Dataset:** `Anthropic/hh-rlhf`
- **Seed:** 42
- **Training source:** official HH-RLHF training split
- **Evaluation source:** official HH-RLHF test split
- **Parsing/filtering:** performed before deterministic sampling
- **Valid train pairs before sampling:** 128,164
- **Valid test pairs before sampling:** 6,767
- **Final training sample:** 5,000 preference pairs
- **Final evaluation sample:** 1,000 preference pairs
- **Maximum prompt length:** 256 tokens
- **Maximum sequence length:** 512 tokens

SFT and DPO originate from the exact same 5,000 underlying training preference pairs. SFT uses the chosen response from each pair as its supervised target. DPO uses the prompt, chosen response, and rejected response from those same pairs. Base, SFT, and DPO are evaluated on the exact same 1,000 held-out test preference pairs.

Evaluation uses mean response-token log-probability. For each pair, the preference margin is:

```text
margin = chosen response score - rejected response score
```

A margin greater than zero counts as a correct preference ranking. Preference accuracy is the proportion of held-out pairs for which the chosen response receives the higher score.

## Training Configuration

| Setting | SFT | DPO |
|---|---:|---:|
| Initialization | Base model | Base model |
| Epochs | 1 | 1 |
| Learning rate | 2e-5 | 1e-6 |
| Per-device batch size | 2 | 1 |
| Gradient accumulation | 8 | 8 |
| Effective batch size | 16 | 8 |
| Optimizer | AdamW | TRL/DPOTrainer default |
| Weight decay | 0.01 | default |
| Scheduler | cosine | default |
| Warmup steps | 20 | default |
| DPO beta | n/a | 0.1 |
| Precision | FP32 | FP32 |
| Optimizer steps | 313 | 625 |
| Training loss | 1.8717 | 0.6896 |
| Runtime | ~16.1 min | ~43.4 min |

SFT and DPO training losses are not directly comparable because they optimize different objectives.

## Final Results

| Model | Correct | Preference Accuracy | Mean Margin | Median Margin |
|---|---:|---:|---:|---:|
| Base | 539/1000 | 53.90% | 0.0629 | 0.0552 |
| SFT | 544/1000 | 54.40% | 0.0611 | 0.0630 |
| DPO | 539/1000 | 53.90% | 0.0664 | 0.0565 |

Paired accuracy analysis on the shared 1,000 examples:

- **SFT - Base:** +0.50 percentage points, 95% paired bootstrap CI [-1.60, +2.60] pp, McNemar p = 0.7044
- **DPO - Base:** 0.00 percentage points, 95% paired bootstrap CI [-0.50, +0.50] pp, McNemar p = 1.0
- **DPO - SFT:** -0.50 percentage points, 95% paired bootstrap CI [-2.60, +1.50] pp, McNemar p = 0.7044

The main paired margin result is:

- **DPO - Base mean paired margin difference:** +0.00351
- **95% paired bootstrap CI:** [0.00219, 0.00488]
- **Paired Wilcoxon signed-rank p-value:** 3.33e-07

SFT shows a small descriptive +0.50 percentage-point accuracy increase over Base, but it is not statistically significant. DPO has the same preference accuracy as Base, while producing a small but systematic positive shift in preference margin relative to Base. Neither method establishes a statistically significant held-out preference-accuracy advantage in this setup.

## Statistical Analysis

The statistical analysis verifies that the Base, SFT, and DPO CSV files contain the same 1,000 `source_index` values in the same order before computing paired statistics. It uses:

- exact two-sided McNemar/binomial tests for paired binary outcomes,
- paired bootstrap confidence intervals,
- 10,000 bootstrap resamples,
- bootstrap seed 42, and
- paired Wilcoxon signed-rank analysis for preference-margin differences.

Relevant files:

- `src/statistical_analysis.py`
- `results/statistical_analysis.json`

## Repository Structure

```text
preference-alignment-sft-vs-dpo/
├── notebooks/
│   ├── SFT.ipynb                 # SFT training and Base/SFT evaluation workflow
│   └── DPO.ipynb                 # DPO training and DPO evaluation workflow
├── results/
│   ├── base_preference_eval.csv  # Per-example Base evaluation
│   ├── sft_preference_eval.csv   # Per-example SFT evaluation
│   ├── dpo_preference_eval.csv   # Per-example DPO evaluation
│   └── statistical_analysis.json # Machine-readable paired statistical summary
├── src/
│   ├── data_preparation.py       # Shared HH-RLHF parsing, filtering, and sampling
│   ├── evaluation.py             # Shared preference-scoring/evaluation functions
│   └── statistical_analysis.py   # Paired statistical analysis script
├── requirements.txt
├── README.md
└── .gitignore
```

## Reproduction / Usage

Install the pinned Python dependencies:

```bash
pip install -r requirements.txt
```

Run the statistical analysis from the repository root:

```bash
python src/statistical_analysis.py
```

The training and evaluation workflows are contained in:

- `notebooks/SFT.ipynb`
- `notebooks/DPO.ipynb`

The notebooks are designed for a GPU-enabled notebook environment such as Google Colab. They intentionally rely on the runtime's PyTorch/CUDA installation rather than installing or upgrading PyTorch themselves.

## Environment

The pinned non-PyTorch dependencies are listed in `requirements.txt`:

```text
transformers==4.53.3
datasets==3.6.0
accelerate==1.8.1
trl==0.19.1
huggingface-hub==0.36.2
fsspec==2025.3.0
```

## Limitations

This is a constrained experiment with one compact base model, one preference dataset, one random seed, and one epoch/configuration per method. The evaluation is likelihood-based preference ranking over existing HH-RLHF response pairs; it is not a human evaluation of newly generated model outputs.
