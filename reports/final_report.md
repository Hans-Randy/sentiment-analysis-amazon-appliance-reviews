# Final Report

## Overview

This project implements a two-phase sentiment analysis experiment on Amazon appliance reviews using a shared, reproducible preprocessing pipeline.

- Phase 1 compares lexicon-based sentiment methods on `data/raw/Appliances_5.json.gz`.
- Phase 2 uses the larger `data/raw/Appliances.json.gz` dataset for preprocessing, exploration, ML baselines, and lexicon comparison.

## Dataset And Labeling

Common preprocessing steps:

- remove rows with empty `reviewText` so every example has usable review content for scoring and feature extraction
- remove exact duplicates by `reviewerID`, `asin`, and `reviewText` to reduce repeated evidence and make the evaluation less biased by duplicated reviews
- combine `summary` and `reviewText` into a single `text` field so short summaries and longer review bodies both contribute to the sentiment signal
- derive sentiment labels from ratings so the sentiment pipeline has a reproducible target variable shared by lexicon and machine-learning comparisons

Label mapping derived from star ratings:

- 1-2 stars -> `Negative`
- 3 stars -> `Neutral`
- 4-5 stars -> `Positive`

## Phase 1: Lexicon Baselines

Phase 1 dataset:

- raw input file: `data/raw/Appliances_5.json.gz`
- rows after empty-text filtering: `2277`
- rows after duplicate removal: `203`

Regenerated results on the prepared dataset:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| VADER | 0.7833 | 0.7972 |
| TextBlob | 0.7488 | 0.7763 |
| SentiWordNet | 0.7635 | 0.8001 |

Phase 1 takeaway:

- VADER and SentiWordNet are the strongest lexicon baselines on the prepared small dataset.
- The refactored SentiWordNet implementation now matches the old notebook behavior much more closely.

## Phase 2: Large-Dataset Workflow

Phase 2 dataset:

- raw input file: `data/raw/Appliances.json.gz`
- rows after empty-text filtering: `602,453`
- rows after duplicate removal: `591,015`
- duplicate-removal share: `1.90%`

Phase 2 exploration:

- the same before/after duplicate-removal exploration is regenerated for the large dataset
- reviewer-bias, reviewer-rating-map, reviewer-word-count, product/reviewer frequency, and duplicate-impact outputs are saved with `phase2_` prefixes under `outputs/figures/` and `outputs/tables/`

Development workflow:

- reproducible stratified development sample from the prepared large dataset: `60,000` rows
- stratified train/test split from the development sample using the raw `overall` rating field: `42,000` train, `18,000` test
- 3-fold cross-validation on the training split
- separate per-model hyperparameter tuning scripts for logistic regression, Linear SVC, and Multinomial Naive Bayes
- lexicon baselines evaluated on a shared stratified `2,000`-review comparison subset from the large-dataset test split
- the final comparison table is aggregated from saved per-model comparison metrics so ML models can be trained independently while preserving the same lexicon comparison subset
- optional experimental models such as MLP and Gradient Boosting can be trained separately without forcing them into every default run

Text representation:

- TF-IDF is used for the machine-learning models because it is a strong and interpretable sparse baseline for sentiment classification, scales well to large review corpora, and works effectively with linear models and Naive Bayes.

Cross-validation summary:

| Model | Mean Weighted F1 |
| --- | ---: |
| Linear SVC | 0.8839 |
| Multinomial Naive Bayes | 0.8731 |
| Logistic Regression | 0.8538 |

Held-out ML test results on the `18,000`-review development test split:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.8932 | 0.8736 |
| Linear SVC | 0.8799 | 0.8840 |
| Logistic Regression | 0.8275 | 0.8534 |

Shared comparison-subset results on the `2,000`-review lexicon comparison subset:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.8915 | 0.8728 |
| Linear SVC | 0.8745 | 0.8801 |
| Logistic Regression | 0.8205 | 0.8494 |
| VADER | 0.7870 | 0.8052 |
| TextBlob | 0.7225 | 0.7574 |
| SentiWordNet | 0.7245 | 0.7527 |

Model evaluation details:

- Accuracy, weighted precision, weighted recall, weighted F1, per-class metrics, and confusion matrices are generated for each machine-learning model.
- The full metric JSON files are saved under `outputs/metrics/`.
- Confusion matrices for the machine-learning models are saved under `outputs/figures/phase2_*_confusion_matrix.png`.
- The assignment minimum is two machine-learning models, but the experiment keeps all three trained models in the final comparison to provide a broader baseline set.

Phase 2 takeaway:

- The larger dataset produces a more realistic Phase 2 experiment than the small Phase 1 file.
- On the shared comparison subset, the ML baselines outperform the lexicon baselines.
- Multinomial Naive Bayes is the strongest current Phase 2 baseline on this development setup, followed by Linear SVC.
- Hyperparameter tuning is now separated from the normal training pipeline so repeated training runs stay faster and easier to reproduce.
- The repository now supports a separate aggregation step so independently trained models and lexicon baselines can still appear in one final comparison table.

## Artifacts

Key outputs are saved under:

- `data/interim/`
- `data/processed/`
- `outputs/metrics/`
- `outputs/predictions/`
- `outputs/tables/`
- `outputs/figures/`
- `outputs/models/`

## Conclusion

The repository now supports a reproducible two-phase workflow with reusable code in `src/`, readable orchestration notebooks, saved experiment artifacts, and basic test coverage. Phase 2 now uses the correct larger dataset and compares lexicon and ML baselines within the same large-dataset evaluation workflow.
