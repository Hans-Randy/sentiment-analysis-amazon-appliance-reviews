# Experiment Notes

- Phase 1 compares three lexicon baselines: VADER, TextBlob, and SentiWordNet.
- Phase 2 starts with simple TF-IDF baselines: logistic regression, Linear SVC, and Multinomial Naive Bayes.
- Labels are derived from ratings with an explicit mapping: 1-2 stars -> Negative, 3 stars -> Neutral, 4-5 stars -> Positive.
- The reproducible preparation pipeline removes exact duplicates by `reviewerID`, `asin`, and `reviewText`, removes empty review text, and combines `summary` with `reviewText` into a single `text` field.
- Generated metrics, predictions, and figures are saved under `outputs/`.
- Phase 1 now also regenerates exploratory figures and tables for both before and after duplicate removal so the experiment can show how deduplication changes the distributions.
- Duplicate removal materially changes the observed distributions: the 3-star share drops from 421 to 9 reviews, while 5-star reviews drop from 1612 to 165, showing that repeated review text heavily inflated the raw counts.
- Current prepared dataset size after duplicate and empty-text filtering: 203 reviews.
- Phase 1 regenerated metrics on the prepared dataset: VADER accuracy `0.7833`, TextBlob accuracy `0.7488`, SentiWordNet accuracy `0.7635`.
- Phase 2 now uses `data/raw/Appliances.json.gz` instead of the small Phase 1 file.
- Large-dataset Phase 2 overview: `602,453` rows after empty-text filtering, `591,015` rows after duplicate removal, then a reproducible `60,000`-row development sample for ML experimentation.
- Phase 2 held-out ML test metrics on the large-dataset development split: Multinomial Naive Bayes accuracy `0.8942`, Linear SVC `0.8792`, Logistic Regression `0.8250`.
- Shared lexicon-vs-ML comparison subset metrics on the large dataset: Multinomial Naive Bayes `0.8940`, Linear SVC `0.8785`, Logistic Regression `0.8400`, VADER `0.7925`, TextBlob `0.7525`, SentiWordNet `0.7435` accuracy.
- The Phase 2 pipeline now also saves a 3-fold cross-validation summary for the ML baselines to `outputs/tables/phase2_cross_validation_summary.csv`.
- Current cross-validation ranking on the large-dataset training split: Linear SVC `0.8788` weighted F1 mean, Multinomial Naive Bayes `0.8701`, Logistic Regression `0.8523`.
- Lightweight hyperparameter search on the large dataset currently favors Linear SVC with `C=0.5` and `tfidf__min_df=1`, and Logistic Regression with `C=2.0` and `tfidf__min_df=1`.
- Phase 2 exploration for the large dataset is saved with `phase2_` prefixes under `outputs/figures/` and `outputs/tables/`.
