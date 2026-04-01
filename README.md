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
│   └── phase2_ml_sentiment.ipynb
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
│   ├── lexicon_baselines.py
│   ├── phase1_exploration.py
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

Run the Phase 2 baseline ML experiment:

```bash
uv run python -m src.train_ml
```

Notes:

- `uv run python -m src.data_prep` prepares the small Phase 1 dataset from `data/raw/Appliances_5.json.gz`
- `uv run python -m src.train_ml` prepares the large Phase 2 dataset from `data/raw/Appliances.json.gz`, saves large-dataset exploration artifacts, trains the ML baselines, and evaluates lexicon baselines on the shared comparison subset

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
- Phase 2 development sample: `60,000` rows; held-out ML test split: `12,000` rows; lexicon comparison subset: `2,000` rows
- Phase 2 held-out ML accuracy: Linear SVC `0.8792`, Multinomial Naive Bayes `0.8942`, Logistic Regression `0.8250`
- Phase 2 shared comparison subset accuracy: Linear SVC `0.8785`, Multinomial Naive Bayes `0.8940`, Logistic Regression `0.8400`, VADER `0.7925`, TextBlob `0.7525`, SentiWordNet `0.7435`
- Phase 2 cross-validation, tuning, exploration, error analysis, and prediction distribution tables are saved under `outputs/tables/` and `outputs/figures/`

## Notebook Role

The notebooks are intentionally lightweight. They orchestrate the reusable code in `src/`, document the workflow, and summarize outputs without duplicating the core pipeline logic.
