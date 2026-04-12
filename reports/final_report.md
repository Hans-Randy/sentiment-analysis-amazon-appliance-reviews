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
- lexicon baselines are tested separately on the full held-out `18,000`-review test split so all Phase 2 models use the exact same evaluation data
- the final comparison table is aggregated afterward from saved machine-learning and lexicon test metrics on that same full shared test set
- optional experimental models such as MLP and Gradient Boosting can be trained separately without forcing them into every default run

Text representation:

- TF-IDF is used for the machine-learning models because it is a strong and interpretable sparse baseline for sentiment classification, scales well to large review corpora, and works effectively with linear models and Naive Bayes.

Cross-validation summary:

| Model | Mean Weighted F1 |
| --- | ---: |
| Linear SVC | 0.8839 |
| Multinomial Naive Bayes | 0.8731 |
| Logistic Regression | 0.8538 |

Held-out ML test results on the shared `18,000`-review development test split:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| MLP | 0.8988 | 0.8848 |
| Multinomial Naive Bayes | 0.8931 | 0.8737 |
| Linear SVC | 0.8824 | 0.8872 |
| Logistic Regression | 0.8273 | 0.8543 |

All-model comparison results on the same shared `18,000`-review Phase 2 test set:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| MLP | 0.8988 | 0.8848 |
| Multinomial Naive Bayes | 0.8931 | 0.8737 |
| Linear SVC | 0.8824 | 0.8872 |
| Logistic Regression | 0.8273 | 0.8543 |
| VADER | 0.7861 | 0.8051 |
| TextBlob | 0.7414 | 0.7718 |
| SentiWordNet | 0.7394 | 0.7649 |

Model evaluation details:

- Accuracy, weighted precision, weighted recall, weighted F1, per-class metrics, and confusion matrices are generated for each machine-learning model.
- The full metric JSON files are saved under `outputs/metrics/`.
- Confusion matrices for the machine-learning models are saved under `outputs/figures/phase2_*_confusion_matrix.png`.
- The assignment minimum is two machine-learning models, but the experiment keeps all three trained models in the final comparison to provide a broader baseline set.

Phase 2 takeaway:

- The larger dataset produces a more realistic Phase 2 experiment than the small Phase 1 file.
- On the full shared test set, the machine-learning baselines outperform the lexicon baselines.
- MLP is the strongest current Phase 2 model on this development setup, followed closely by Multinomial Naive Bayes and Linear SVC.
- Hyperparameter tuning is now separated from the normal training pipeline so repeated training runs stay faster and easier to reproduce.
- The repository now supports a separate lexicon-testing step and a separate aggregation step, so independently trained ML models and tested lexicon baselines can still appear in one final comparison table.

## Artifacts

Key outputs are saved under:

- `data/interim/`
- `data/processed/`
- `outputs/metrics/`
- `outputs/predictions/`
- `outputs/tables/`
- `outputs/figures/`
- `outputs/models/`

## Section 15: Rating Enhancement

Method used:

- overall-opinion-enhanced rating blend based on review text and the strongest existing sentiment model already trained in the repo

Why this method was chosen:

- it matches the project brief's simplified paper-inspired requirement directly
- it reuses the existing sentiment infrastructure already built for the COMP 262 project
- it is easier to justify and implement clearly than topic, helpfulness, or aspect-based alternatives that are not already part of this repo

Implementation summary:

- source dataset: `data/processed/amazon_appliances_large_reviews_labeled.csv`
- usable rows: `591,015`
- text field used: merged `text = summary + reviewText`
- inferred overall opinion source: `outputs/models/phase2_mlp.joblib`
- sentiment-to-rating mapping:
  - Negative -> `1.5`
  - Neutral -> `3.0`
  - Positive -> `4.5`
- enhanced rating formula:
  - `enhanced_rating = alpha * original_rating + (1 - alpha) * inferred_rating`

Alpha results:

| Alpha | MAE | RMSE |
| --- | ---: | ---: |
| 0.9 | 0.0667 | 0.0810 |
| 0.8 | 0.1333 | 0.1621 |
| 0.7 | 0.2000 | 0.2431 |
| 0.6 | 0.2667 | 0.3241 |

Section 15 takeaway:

- the best setting on this internal evaluation is `alpha = 0.9`
- because the original rating is used as the evaluation reference, the strongest result naturally stays close to the original rating
- the section still demonstrates a complete and reproducible implementation of review-based rating enhancement

Detailed write-up:

- `reports/section15_rating_enhancement.md`

## Section 16: Local Review Summarization

Task setup:

- selected exactly `10` reviews with `review_word_count > 100`
- selection is deterministic
- local Hugging Face summarization model used: `sshleifer/distilbart-cnn-12-6`

Why this model was used:

- it is small enough to run locally on CPU hardware
- it is specifically designed for summarization-style generation
- it avoids dependence on hosted APIs

Generation setup:

- deterministic decoding (`do_sample=False`)
- target length: about `50` words
- `min_new_tokens = 45`
- `max_new_tokens = 96`

Observed result summary:

- summary lengths ranged from `44` to `71` words for most examples, with one longer outlier at `81`
- the model usually preserves the main product opinion and the central issue or benefit
- some summaries still contain repeated phrases or awkward trailing text, which is a limitation of the compact CPU-friendly local model

Detailed write-up:

- `reports/section16_llm_summarization.md`

## Section 17: Local Customer-Service Response

Task setup:

- selected one deterministic question-like review using simple heuristics on the merged `text` field
- selected review index: `287943`
- selected because it contains a question mark and asks whether the cracked lid window should be returned or replaced locally
- local Hugging Face response model used: `Qwen/Qwen2.5-0.5B-Instruct`

Prompt strategy:

- few-shot instruction prompt
- asks for a polite, concise customer-service style response
- encourages acknowledgement of the issue and a safe next step without making guarantees

Observed quality:

- the generated response is polite and broadly customer-service oriented
- however, it remains generic and is not fully grounded in the exact cracked-window scenario
- this is acceptable as a local CPU-friendly baseline, but the response would still need human review before real deployment

Detailed write-up:

- `reports/section17_llm_response.md`

## Conclusion

The repository now supports a reproducible two-phase workflow with reusable code in `src/`, readable orchestration notebooks, saved experiment artifacts, and basic test coverage. Phase 2 now uses the correct larger dataset, compares lexicon and ML baselines within the same large-dataset evaluation workflow, implements review-based rating enhancement, and adds locally hosted Hugging Face workflows for review summarization and customer-service response generation.
