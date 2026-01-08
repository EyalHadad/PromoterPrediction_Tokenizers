# Optimizing Genomic Language Models for Promoter Prediction

This repository contains the complete code for reproducing the experiments described in the paper: **"Optimizing Genomic Language Models for Promoter Prediction: A Comparative Study of Tokenization and Cross-Species Learning."**

The experiments cover four tokenization methods, two negative data generation strategies, pretraining validation, cross-species transfer learning, and Explainable AI (XAI) analysis using SHAP.

## ⚙️ Environment Setup

[cite_start]This project relies on **Python** and **Conda** for environment management to ensure reproducibility (as detailed in the "Availability of data and materials" section [cite: 1164]).

1.  **Install Conda:** Ensure you have a working installation of Anaconda or Miniconda.
2.  [cite_start]**Create and Activate the Environment:** Use the provided environment file (`PromoterPrediction.yml` - **Note:** This file is expected to be present in the final repository, as promised in the response to reviewers [cite: 66, 67]):
    ```bash
    conda env create -f PromoterPrediction.yml
    conda activate promoter_prediction
    ```

## 📂 Repository Structure (Required)

Before running the scripts, the following directory structure and data files must be in place.

| Directory/File | Purpose |
| :--- | :--- |
| `data/` | **Input Data:** Must contain the processed train/val/test CSV files for both negative data strategies (e.g., `data/Substitution/train_human.csv`). Requires raw FASTA/BED files for initial processing. |
| `tokenizers/` | Stores the trained BPE, WPC, and k-mer tokenizer files (`*.json`). |
| `pretrain_models/` | Stores the RoBERTa models after the Masked Language Modeling (MLM) stage. |
| `finetune_models/` | Stores the final, organism-specific fine-tuned models. |
| `results/` | Stores raw prediction files and intermediate statistical results (used by `model_delong.py`, `model_evaluation.py`). |
| `evaluation/` | Stores the final calculated performance metrics (AUC, F1, MCC). |
| `delong_comparisons/` | Stores the CSV results and heatmaps from the DeLong statistical significance tests. |
| `shap_plots/` | Stores raw SHAP value outputs and all generated SHAP figures (Figures 3, S4, S6). |
| `logs/` | Stores training and fine-tuning log files. |

## 🚀 Workflow: Step-by-Step Reproduction

The complete reproduction requires running the scripts in the specified order.

### 1. Data Preparation and Tokenizer Training

This stage generates the datasets and the necessary tokenizers.

| Script | Purpose | Output Location |
| :--- | :--- | :--- |
| `train_val_test_split.py` | Creates the train, validation and test sets from CDHIT output. | `train_ids.csv` `val_ids.csv` `test_ids.csv` |
| `positive_data_creation.py` | Converts raw FASTA data to CSV format. | `data/positive_data.csv` |
| `negative_data_substitution.py` | Generates the **Positive-promoter-shuffled** dataset (Substitution method). | `data/Substitution/` |
| `negative_data_genome.py` | Generates the **Random-non-promoter-fragments** dataset (Genome method). | `data/Genome/` |
| `tokenizer_training.py` | Trains the four required tokenizers (k-mer non-overlapping/overlapping, BPE, WPC). | `tokenizers/` |

### 2. Base Model Training (Pretraining)

[cite_start]This stage performs Masked Language Modeling (MLM) exclusively on the **positive promoter sequences**[cite: 21].

*The default `model_pretraining.py` covers the "Flat Pretrain" approach.*

| Script | Purpose | Output Location |
| :--- | :--- | :--- |
| `model_pretraining.py` | Trains RoBERTaForMaskedLM for all four tokenizers on the combined positive training data. | `pretrain_models/` |

### 3. Fine-Tuning and Core Evaluation

This stage fine-tunes the pretrained models for classification (using both positive and negative samples) and evaluates their performance (Figures 2, 5, S3).

| Script | Purpose | Output Location |
| :--- | :--- | :--- |
| `model_finetune.py` | Performs fine-tuning for all 32 models (4 tokenizers $\times$ 8 organisms $\times$ 2 negative methods). | `finetune_models/` |
| `model_prediction.py` | Generates probability predictions for all test sets using the fine-tuned models. | `results/` |
| `model_evaluation.py` | Calculates core performance metrics (AUC, F1, MCC) for all models. | `evaluation/` |
| `model_delong.py` | [cite_start]Performs the **DeLong Test** with Benjamini-Hochberg (FDR) correction (Figure S3)[cite: 164]. | `delong_comparisons/` |

### 4. Explainable AI (XAI) Analysis

This stage generates the position-wise and individual SHAP importance plots (Figures 3, S4, S6).

*Requires the raw SHAP value CSVs to be present in the `shap_plots/` directory.*

| Script | Purpose | Output Location |
| :--- | :--- | :--- |
| `shap_importance_locations_plots.py` | Generates position-wise average SHAP importance plots (Figure 3, Figure S4). | `shap_plots/shap_importance_plots/` |
| `shap_plots.py` | Contains helper functions for plotting the individual top 5 sequences (Figure S6) and nucleotide distribution analysis (Figure S5). | `shap_plots/shap_importance_plots/` |

### 5. Advanced Experiments Reproduction 

To facilitate the reproduction of the additional control experiments requested during the review process, we provide a unified automation script: `run_advanced_experiments.py`. This script handles data preprocessing (including leakage prevention), model initialization, and evaluation for the following scenarios.

#### A. External Validation on *C. familiaris* (Dog) [Figure 5]
[cite_start]Performs real-time filtering of dog promoter sequences against the training set of all other organisms using CD-HIT (80% identity threshold) to ensure strictly non-redundant external validation [cite: 169-170, 242-244].
* **Prerequisite:** Ensure `data/C.familiaris.txt` (FASTA format) is present.
* **Command:**
    ```bash
    python run_advanced_experiments.py --experiment dog
    ```

#### B. Character-Level Tokenizer [Figure S7]
[cite_start]Trains a character-level tokenizer (treating each nucleotide as a token), performs pretraining in-memory, and fine-tunes the model to evaluate the efficacy of single-nucleotide tokens versus motif-based tokens [cite: 425-427].
* **Command:**
    ```bash
    python run_advanced_experiments.py --experiment char
    ```

#### C. Impact of Promoter Length (300bp) [Figure S9]
Evaluates model performance when the input sequence is restricted to a shorter window (-250 to +50 bp relative to TSS). [cite_start]The script automatically center-crops the 600bp sequences during data loading [cite: 485-486].
* **Command:**
    ```bash
    python run_advanced_experiments.py --experiment short
    ```

#### D. No-Pretrain Baseline [Figure S8]
[cite_start]Trains the classification model directly on the labeled fine-tuning data with randomly initialized weights, skipping the Masked Language Modeling (MLM) phase, to quantify the contribution of pretraining [cite: 476-478].
* **Command:**
    ```bash
    python run_advanced_experiments.py --experiment no_pretrain
    ```