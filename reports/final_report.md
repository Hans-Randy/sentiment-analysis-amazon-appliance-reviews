# Final Report

## Overview

This project implements a two-phase sentiment analysis experiment on Amazon appliance reviews using a shared, reproducible preprocessing pipeline.

- Phase 1 compares lexicon-based sentiment methods on `data/raw/Appliances_5.json.gz`.
- Phase 2 uses the larger `data/raw/Appliances.json.gz` dataset for preprocessing, exploration, ML baselines, and lexicon comparison.

## Dataset And Labeling

Common preprocessing steps:

- remove rows with empty `reviewText`
- remove exact duplicates by `reviewerID`, `asin`, and `reviewText`
- combine `summary` and `reviewText` into a single `text` field

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
- stratified train/test split from the development sample: `48,000` train, `12,000` test
- 3-fold cross-validation on the training split
- lightweight hyperparameter search for logistic regression and Linear SVC
- lexicon baselines evaluated on a shared stratified `2,000`-review comparison subset from the large-dataset test split

Cross-validation summary:

| Model | Mean Weighted F1 |
| --- | ---: |
| Linear SVC | 0.8788 |
| Multinomial Naive Bayes | 0.8701 |
| Logistic Regression | 0.8523 |

Held-out ML test results on the `12,000`-review development test split:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.8942 | 0.8739 |
| Linear SVC | 0.8792 | 0.8832 |
| Logistic Regression | 0.8250 | 0.8521 |

Shared comparison-subset results on the `2,000`-review lexicon comparison subset:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.8940 | 0.8737 |
| Linear SVC | 0.8785 | 0.8810 |
| Logistic Regression | 0.8400 | 0.8621 |
| VADER | 0.7925 | 0.8075 |
| TextBlob | 0.7525 | 0.7881 |
| SentiWordNet | 0.7435 | 0.7673 |

Phase 2 takeaway:

- The larger dataset produces a more realistic Phase 2 experiment than the small Phase 1 file.
- On the shared comparison subset, the ML baselines outperform the lexicon baselines.
- Multinomial Naive Bayes is the strongest current Phase 2 baseline on this development setup, followed by Linear SVC.

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
