# Sentiment Analysis on Amazon Appliance Reviews

This repository contains a two-phase sentiment analysis experiment on Amazon appliance reviews.

## Phases

- Phase 1 compares three lexicon-based sentiment methods: VADER, TextBlob, and SentiWordNet.
- Phase 2 uses the larger `data/raw/Appliances.json.gz` dataset, performs the same exploration workflow, and compares TF-IDF machine-learning baselines with lexicon baselines on a shared evaluation subset.

## Project Layout

```text
sentiment_analysis_amazon_appliance_reviews/
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── raw/
│   │   ├── Appliances_5.json.gz
│   │   └── Appliances.json.gz
│   ├── interim/
│   │   ├── amazon_appliances_reviews_prepared.csv
│   │   ├── amazon_appliances_large_reviews_prepared.csv
│   │   └── amazon_appliances_large_phase2_development_sample_prepared.csv
│   └── processed/
│       ├── amazon_appliances_reviews_labeled.csv
│       ├── amazon_appliances_large_reviews_labeled.csv
│       └── amazon_appliances_large_phase2_development_sample_labeled.csv
├── notebooks/
│   ├── old_phase1_lexicon_comparison.ipynb
│   ├── phase1_lexicon_comparison.ipynb
│   ├── phase2_compare_lexicons.ipynb
│   ├── phase2_compare_models.ipynb
│   ├── phase2_prepare.ipynb
│   ├── phase2_train_default_models.ipynb
│   ├── phase2_train_gradient_boosting.ipynb
│   ├── phase2_train_mlp.ipynb
│   └── phase2_workflow_index.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── tables/
├── reports/
│   ├── experiment_notes.md
│   └── final_report.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── features.py
│   ├── compare_lexicons.py
│   ├── compare_models.py
│   ├── lexicon_baselines.py
│   ├── llm_review_response.py
│   ├── llm_summarize_reviews.py
│   ├── model_registry.py
│   ├── phase1_exploration.py
│   ├── prepare_phase2.py
│   ├── rating_enhancement.py
│   ├── tune_gradient_boosting.py
│   ├── tune_logistic_regression.py
│   ├── tune_mlp.py
│   ├── tune_multinomial_nb.py
│   ├── tune_svm.py
│   ├── tune_utils.py
│   ├── train_ml.py
│   └── utils.py
└── tests/
    └── test_data_pipeline.py
```

## Label Mapping

Labels are derived directly from star ratings:

- 1-2 stars -> `Negative`
- 3 stars -> `Neutral`
- 4-5 stars -> `Positive`

This mapping is implemented in `src/data_prep.py` and reused across both phases.

## Reproducible Preprocessing

The shared preparation pipeline:

- loads the requested raw dataset file explicitly for each phase
- removes empty `reviewText` rows
- removes exact duplicates by `reviewerID`, `asin`, and `reviewText`
- combines `summary` and `reviewText` into a single `text` feature
- saves prepared data to `data/interim/` and `data/processed/`

## Environment

Use `uv` for dependency management and command execution.

Install dependencies:

```bash
uv sync
```

## Run The Pipelines

Prepare the Phase 1 dataset helper output:

```bash
uv run python -m src.data_prep
```

Run Phase 1 lexicon baselines:

```bash
uv run python -m src.lexicon_baselines
```

Prepare the Phase 2 large-dataset artifacts:

```bash
uv run python -m src.prepare_phase2
```

Useful Phase 2 preparation overrides:

```bash
uv run python -m src.prepare_phase2 --sample-size 60000 --comparison-size 2000
uv run python -m src.prepare_phase2 --sample-size 100000 --comparison-size 5000 --skip-exploration
```

Tune Phase 2 models when needed:

```bash
uv run python -m src.tune_logistic_regression
uv run python -m src.tune_multinomial_nb
uv run python -m src.tune_svm
uv run python -m src.tune_mlp
uv run python -m src.tune_gradient_boosting
```

Run the shared lexicon comparison and aggregate all results:

```bash
uv run python -m src.compare_lexicons
uv run python -m src.compare_models
```

Run the remaining COMP 262 Phase 2 Sections 15-17 deliverables:

```bash
uv run python -m src.rating_enhancement
uv run python -m src.llm_summarize_reviews
uv run python -m src.llm_review_response
```

Run the Phase 2 baseline ML experiment with fixed defaults:

```bash
uv run python -m src.train_ml
```

Optional selective training examples:

```bash
uv run python -m src.train_ml --models logistic_regression svm multinomial_nb
uv run python -m src.train_ml --models mlp --skip-lexicon
uv run python -m src.train_ml --include-experimental --skip-lexicon
uv run python -m src.train_ml --models svm multinomial_nb --skip-cv
```

Notes:

- `uv run python -m src.data_prep` prepares the small Phase 1 dataset from `data/raw/Appliances_5.json.gz`
- `uv run python -m src.prepare_phase2` prepares the large Phase 2 dataset from `data/raw/Appliances.json.gz` and saves the large-dataset exploration artifacts
- tuning is now separated from training; review the tuning outputs, then manually promote the chosen parameters into `src.train_ml.py`
- `uv run python -m src.train_ml` trains and evaluates the Phase 2 baselines using the fixed defaults in `src.train_ml.py`
- Phase 2 now uses a 70/30 train/test split stratified by the raw `overall` rating field
- `src.model_registry.py` defines default and experimental model pipelines plus their tuning grids
- experimental models (`mlp`, `gradient_boosting`) are available through selective training and are not part of the default run
- `src.prepare_phase2.py` now accepts `--sample-size`, `--comparison-size`, `--raw-data-path`, and `--skip-exploration`
- `src.train_ml.py` now accepts `--skip-cv`, `--test-size`, `--prepared-sample-path`, and `--comparison-subset-path`
- `uv run python -m src.compare_lexicons` evaluates VADER, TextBlob, and SentiWordNet on the saved shared Phase 2 comparison subset
- `uv run python -m src.compare_models` merges saved ML and lexicon comparison metrics into the final comparison table and figure
- `uv run python -m src.rating_enhancement` implements the paper-inspired review-based rating enhancement experiment for Section 15
- `uv run python -m src.llm_summarize_reviews` generates 10 local-Hugging-Face summaries for Section 16
- `uv run python -m src.llm_review_response` generates one local-Hugging-Face customer-service style response for Section 17

Run tests:

```bash
uv run python -m pytest tests/test_data_pipeline.py
```

## Generated Artifacts

- Prepared datasets: `data/interim/`, `data/processed/`
- Metrics JSON: `outputs/metrics/`
- Prediction CSV files: `outputs/predictions/`
- Summary tables: `outputs/tables/`
- Confusion matrix figures: `outputs/figures/`
- Trained baseline model: `outputs/models/`
- Phase 1 and Phase 2 exploration figures include before/after duplicate-removal comparisons for ratings, labels, review lengths, reviewer/product frequency, missing values, reviewer-bias views, and duplicate-impact plots

## Current Baseline Snapshot

- Prepared dataset after duplicate and empty-text filtering: `203` reviews
- Phase 1 regenerated accuracy: VADER `0.7833`, TextBlob `0.7488`, SentiWordNet `0.7635`
- Phase 2 source dataset: `data/raw/Appliances.json.gz` with `602,453` rows after empty-text filtering and `591,015` rows after duplicate removal
- Phase 2 development sample: `60,000` rows; held-out ML test split: `18,000` rows; lexicon comparison subset: `2,000` rows
- Phase 2 uses TF-IDF because it is a strong, interpretable sparse baseline for large review corpora and works well with linear models and Naive Bayes
- Phase 2 held-out ML accuracy: Linear SVC `0.8799`, Multinomial Naive Bayes `0.8932`, Logistic Regression `0.8275`
- Phase 2 shared comparison subset accuracy: Linear SVC `0.8745`, Multinomial Naive Bayes `0.8915`, Logistic Regression `0.8205`, VADER `0.7870`, TextBlob `0.7225`, SentiWordNet `0.7245`
- Phase 2 cross-validation, exploration, error analysis, and prediction distribution tables are saved under `outputs/tables/` and `outputs/figures/`
- Per-model tuning outputs are saved as `outputs/tables/tuning_*.csv` and `outputs/metrics/tuning_*.json`
- Final all-model comparison is built from saved shared-subset metrics so ML models can be trained separately while lexicons remain in the same comparison
- Section 15 artifacts are saved as `outputs/tables/rating_enhancement_*.csv` and `reports/section15_rating_enhancement.md`
- Section 16 artifacts are saved as `outputs/tables/section16_review_summaries.csv` and `reports/section16_llm_summarization.md`
- Section 17 artifacts are saved as `outputs/tables/section17_customer_response.csv` and `reports/section17_llm_response.md`

## Colab-Friendly Usage

- The Phase 2 scripts are designed to remain local-first while also working well in Google Colab.
- Recommended Colab pattern:
  1. mount Google Drive
  2. place `Appliances.json.gz` in a Drive-backed folder
  3. set env vars such as `PHASE2_DATA_ROOT`, `PHASE2_OUTPUT_ROOT`, `PHASE2_MODELS_ROOT`, and `HF_CACHE_DIR`
  4. run `uv run python -m src.prepare_phase2 --sample-size <n> --comparison-size <m>`
  5. run `uv run python -m src.train_ml --models ... --skip-cv` for heavier experiments
- Hugging Face helpers automatically use CUDA when available and fall back to CPU otherwise.
- `gradient_boosting` remains the slowest experimental model even in stronger environments, so it is best treated as an optional run rather than a default baseline.

## Notebook Role

The notebooks are intentionally lightweight. They follow the CLI workflow step-by-step, document each stage of the experiment, and summarize saved outputs without duplicating the core pipeline logic.
