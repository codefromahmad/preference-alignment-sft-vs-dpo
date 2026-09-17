# Preference Alignment: SFT vs DPO

This repository contains the experimental code for comparing **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** for preference alignment of a compact language model under limited training data.

## Research Question

> How do Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) compare in aligning a compact language model with human preferences under limited training data?

## Experimental Setup

- **Base model:** `HuggingFaceTB/SmolLM2-360M`
- **Dataset:** Anthropic HH-RLHF
- **Training data:** 1,000 preference pairs
- **Training epochs:** 1
- **Random seed:** 42
- **Methods:** SFT and DPO
- **Evaluation:** Preference accuracy based on chosen vs. rejected responses

## Repository Structure

```text
preference-alignment-sft-vs-dpo/
├── notebooks/
│   ├── SFT.ipynb
│   └── DPO.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Notebooks

### `notebooks/SFT.ipynb`

Fine-tunes SmolLM2-360M on preferred responses from HH-RLHF preference pairs and evaluates whether the resulting model assigns higher scores to chosen responses than rejected responses.

### `notebooks/DPO.ipynb`

Applies Direct Preference Optimization to the same compact base model using chosen/rejected response pairs and evaluates held-out preference discrimination.

## Results

| Model | Preference Accuracy | Correct / Evaluated |
|---|---:|---:|
| Base model | 51.53% | 101 / 196 |
| SFT | 57.14% | 112 / 196 |
| DPO | 52.76% | 105 / 199 |

SFT achieved the highest preference accuracy in this experimental setup. The DPO evaluation contains 199 held-out pairs, while the Base and SFT evaluations contain 196, so comparisons involving DPO should be interpreted descriptively rather than as results on an identical evaluation sample.

These results are specific to the compact model, limited-data setting, and hyperparameters used here; they are not intended as a universal ranking of SFT and DPO.

## Environment

The experiments were run in Google Colab with a CUDA-enabled GPU. The main Python dependencies are pinned in `requirements.txt`.

The notebooks intentionally use the PyTorch installation provided by the runtime rather than installing or upgrading PyTorch separately.
