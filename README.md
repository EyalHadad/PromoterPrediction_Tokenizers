# PromoterPrediction_Tokenizers

This repository contains the full pipeline for evaluating tokenization methods in genomic promoter classification using large language models (LLMs). It includes data preparation, tokenizer training, model pretraining, fine-tuning, evaluation, and explainability.

## Project Overview

The project compares four tokenization strategies—non-overlapping 6-mer, overlapping 6-mer, BPE, and WPC—across multiple organisms and two negative data generation methods. The pipeline trains and evaluates models using standard classification metrics and includes tools for statistical testing and interpretability analysis.

## Pipeline Structure

1. **Data Creation**
   - `positive_data_creation.py`: Creates positive promoter sequences from EPD.
   - `negative_data_substitution.py`: Generates negative sequences by shuffling (positive-promoter-shuffled).
   - `negative_data_genome.py`: Generates random negative fragments from the genome (random-non-promoter-fragments).

2. **Tokenization**
   - `tokenizer_training.py`: Trains BPE, WPC, and k-mer tokenizers.

3. **Pretraining**
   - `model_pretraining.py`: Pretrains a masked language model on tokenized positive sequences.

4. **Fine-Tuning**
   - `model_finetune.py`: Fine-tunes models using labeled data for each organism.

5. **Prediction**
   - `model_prediction.py`: Applies models to the test set.

6. **Evaluation**
   - `model_evaluation.py`: Computes ACC, F1, AUC, MCC, TP, and FP for all models.

7. **Model Comparison**
   - `model_delong.py`: Performs pairwise DeLong tests to assess statistical differences between models.

8. **Explainability**
   - `shap_importance_plots.py`: Generates SHAP-based visualizations for token importance across positions.

## Directory Structure

```
PromoterPrediction_Tokenizers/
├── data/                     # Input sequences (positive and two negative types)
├── delong_comparisons/       # Outputs from DeLong comparisons
├── evaluation/               # Summary metrics for each model
├── finetune_models/          # Fine-tuned models
├── logs/                     # Logs from runs
├── plots/                    # Visualizations and SHAP plots
├── pretrain_models/          # Pretrained masked language models
├── results/                  # Model prediction outputs
├── tokenizers/               # Trained tokenizer models
├── EDA.py
├── positive_data_creation.py
├── negative_data_substitution.py
├── negative_data_genome.py
├── tokenizer_training.py
├── model_pretraining.py
├── model_finetune.py
├── model_prediction.py
├── model_evaluation.py
├── model_delong.py
├── shap_importance_plots.py
├── README.md
└── .gitignore
```

## Notes

- Each organism has one positive dataset and two negative datasets:
  - `substitution`: represents the positive-promoter-shuffled method.
  - `genome`: represents the random-non-promoter-fragments method.
- File suffixes indicate the method used:
  - Files ending in `substitution` correspond to the shuffled negative dataset.
  - Files ending in `genome` correspond to the random-fragment negative dataset.