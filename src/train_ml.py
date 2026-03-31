from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    DEFAULT_RANDOM_STATE,
    FIGURES_DIR,
    LARGE_RAW_REVIEW_FILE,
    METRICS_DIR,
    PHASE2_DEV_SAMPLE_SIZE,
    PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE,
    PREDICTIONS_DIR,
    TABLES_DIR,
)
from src.data_prep import (
    load_amazon_reviews,
    prepare_and_save_dataset,
    prepare_dataset,
    resolve_large_raw_data_path,
    save_prepared_dataset,
)
from src.evaluate import (
    compute_classification_metrics,
    metrics_row,
    save_confusion_matrix,
)
from src.features import build_tfidf_vectorizer
from src.lexicon_baselines import run_lexicon_models
from src.phase1_exploration import save_dataset_exploration_outputs
from src.utils import ensure_directories, write_json


def build_model_pipelines() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer().set_params(min_df=2)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        C=2.0,
                        class_weight="balanced",
                        random_state=DEFAULT_RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer().set_params(min_df=2)),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=DEFAULT_RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MultinomialNB": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer()),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def model_param_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "LogisticRegression": {
            "tfidf__min_df": [1, 2],
            "classifier__C": [0.5, 1.0, 2.0],
        },
        "LinearSVC": {
            "tfidf__min_df": [1, 2],
            "classifier__C": [0.5, 1.0, 2.0],
        },
    }


def build_phase2_dataset() -> tuple[
    pd.DataFrame, pd.DataFrame, dict, tuple[Path, Path]
]:
    source_path = resolve_large_raw_data_path()
    raw_loaded_df = load_amazon_reviews(source_path)
    raw_prepared_df = prepare_dataset(raw_loaded_df, remove_exact_duplicates=False)
    prepared_df = prepare_dataset(raw_loaded_df, remove_exact_duplicates=True)
    profile = {
        "source_file": str(
            LARGE_RAW_REVIEW_FILE.relative_to(LARGE_RAW_REVIEW_FILE.parents[2])
        ),
        "raw_rows_after_empty_text_filter": int(len(raw_prepared_df)),
        "prepared_rows": int(len(prepared_df)),
        "duplicate_rows_removed": int(len(raw_prepared_df) - len(prepared_df)),
        "rating_distribution": {
            str(key): int(value)
            for key, value in prepared_df["overall"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "label_distribution": {
            key: int(value)
            for key, value in prepared_df["label"].value_counts().to_dict().items()
        },
        "label_mapping": {"1-2": "Negative", "3": "Neutral", "4-5": "Positive"},
        "duplicate_policy": "Drop exact duplicates by reviewerID, asin, and reviewText.",
        "random_state": DEFAULT_RANDOM_STATE,
    }
    saved_paths = save_prepared_dataset(
        prepared_df, dataset_name="amazon_appliances_large_reviews"
    )
    return raw_prepared_df, prepared_df, profile, saved_paths


def build_development_sample(prepared_df: pd.DataFrame) -> pd.DataFrame:
    if len(prepared_df) <= PHASE2_DEV_SAMPLE_SIZE:
        return prepared_df.reset_index(drop=True)
    development_df, _ = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            prepared_df,
            train_size=PHASE2_DEV_SAMPLE_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=prepared_df["label"],
        ),
    )
    return development_df.reset_index(drop=True)


def build_lexicon_comparison_subset(test_df: pd.DataFrame) -> pd.DataFrame:
    if len(test_df) <= PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE:
        return test_df.reset_index(drop=True)
    lexicon_subset_df, _ = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            test_df,
            train_size=PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=test_df["label"],
        ),
    )
    return lexicon_subset_df.reset_index(drop=True)


def cross_validate_models(train_df: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE
    )
    rows = []
    train_text = cast(pd.Series, train_df["text"])
    train_label = cast(pd.Series, train_df["label"])

    for model_name, pipeline in build_model_pipelines().items():
        scores = cross_validate(
            pipeline,
            train_text,
            train_label,
            cv=splitter,
            scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
            n_jobs=None,
        )
        rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_accuracy_std": float(scores["test_accuracy"].std()),
                "cv_f1_weighted_mean": float(scores["test_f1_weighted"].mean()),
                "cv_f1_weighted_std": float(scores["test_f1_weighted"].std()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["cv_f1_weighted_mean", "cv_accuracy_mean"], ascending=False
    )


def tune_top_models(train_df: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE
    )
    train_text = cast(pd.Series, train_df["text"])
    train_label = cast(pd.Series, train_df["label"])
    rows = []

    for model_name, param_grid in model_param_grids().items():
        search = GridSearchCV(
            estimator=build_model_pipelines()[model_name],
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=splitter,
            n_jobs=None,
            refit=True,
        )
        search.fit(train_text, train_label)
        rows.append(
            {
                "model": model_name,
                "best_cv_f1_weighted": float(search.best_score_),
                "best_params": str(search.best_params_),
            }
        )

    return pd.DataFrame(rows).sort_values(by="best_cv_f1_weighted", ascending=False)


def build_error_tables(
    prediction_frame: pd.DataFrame, prediction_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_distribution = (
        prediction_frame.groupby(["label", prediction_column])
        .size()
        .to_frame("count")
        .reset_index()
    )
    errors = prediction_frame.loc[
        prediction_frame["label"] != prediction_frame[prediction_column]
    ].copy()
    errors["text_preview"] = errors["text"].str.slice(0, 160)
    error_table = errors[
        ["label", prediction_column, "overall", "summary", "text_preview"]
    ].reset_index(drop=True)
    return class_distribution, error_table


def evaluate_ml_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, Pipeline]]:
    model_dir = Path("outputs") / "models"
    ensure_directories([model_dir])

    prediction_frames = []
    metrics_by_model = {}
    fitted_models: dict[str, Pipeline] = {}
    summary_rows = []
    train_text = cast(pd.Series, train_df["text"])
    train_label = cast(pd.Series, train_df["label"])
    test_text = cast(pd.Series, test_df["text"])
    test_label = cast(pd.Series, test_df["label"])

    for model_name, pipeline in build_model_pipelines().items():
        fitted_pipeline = pipeline.fit(train_text, train_label)
        fitted_models[model_name] = fitted_pipeline
        predictions = pd.Series(fitted_pipeline.predict(test_text), index=test_df.index)
        metrics = compute_classification_metrics(test_label, predictions)
        metrics_by_model[model_name] = metrics
        summary_rows.append(metrics_row(model_name, metrics))

        prediction_frame = cast(
            pd.DataFrame, test_df[["overall", "summary", "reviewText", "text", "label"]]
        ).copy()
        prediction_column = f"{model_name}_pred"
        prediction_frame[prediction_column] = predictions.values
        prediction_frame["model"] = model_name
        prediction_frames.append(prediction_frame)

        prediction_frame.to_csv(
            PREDICTIONS_DIR / f"phase2_{model_name.lower()}_test_predictions.csv",
            index=False,
        )
        write_json(metrics, METRICS_DIR / f"phase2_{model_name.lower()}_metrics.json")
        save_confusion_matrix(
            test_label,
            predictions,
            f"Phase 2 {model_name} Confusion Matrix",
            FIGURES_DIR / f"phase2_{model_name.lower()}_confusion_matrix.png",
        )
        joblib.dump(fitted_pipeline, model_dir / f"phase2_{model_name.lower()}.joblib")

        class_distribution, error_table = build_error_tables(
            prediction_frame, prediction_column
        )
        class_distribution.to_csv(
            TABLES_DIR / f"phase2_{model_name.lower()}_class_distribution.csv",
            index=False,
        )
        error_table.to_csv(
            TABLES_DIR / f"phase2_{model_name.lower()}_error_analysis.csv", index=False
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["f1_weighted", "accuracy"], ascending=False
    )
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    return summary_df, combined_predictions, metrics_by_model, fitted_models


def evaluate_ml_on_subset(
    fitted_models: dict[str, Pipeline], subset_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    subset_text = cast(pd.Series, subset_df["text"])
    subset_label = cast(pd.Series, subset_df["label"])
    for model_name, pipeline in fitted_models.items():
        predictions = pd.Series(pipeline.predict(subset_text), index=subset_df.index)
        rows.append(
            metrics_row(
                model_name, compute_classification_metrics(subset_label, predictions)
            )
        )
    return pd.DataFrame(rows)


def run_ml_pipeline() -> dict:
    ensure_directories([FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR])

    raw_prepared_df, prepared_df, profile, saved_paths = build_phase2_dataset()
    save_dataset_exploration_outputs(raw_prepared_df, prepared_df, prefix="phase2")

    development_df = build_development_sample(prepared_df)
    development_paths = save_prepared_dataset(
        development_df,
        dataset_name="amazon_appliances_large_phase2_development_sample",
    )

    train_df, test_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            development_df,
            test_size=0.2,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=development_df["label"],
        ),
    )

    cv_summary_df = cross_validate_models(train_df)
    cv_summary_df.to_csv(
        TABLES_DIR / "phase2_cross_validation_summary.csv", index=False
    )

    tuning_df = tune_top_models(train_df)
    tuning_df.to_csv(TABLES_DIR / "phase2_hyperparameter_search.csv", index=False)

    ml_summary_df, ml_predictions_df, ml_metrics, fitted_models = evaluate_ml_models(
        train_df, test_df
    )
    ml_summary_df.to_csv(TABLES_DIR / "phase2_ml_model_summary.csv", index=False)
    ml_predictions_df.to_csv(
        PREDICTIONS_DIR / "phase2_ml_all_test_predictions.csv", index=False
    )

    lexicon_subset_df = build_lexicon_comparison_subset(test_df)
    ml_subset_summary_df = evaluate_ml_on_subset(fitted_models, lexicon_subset_df)
    lexicon_test_results, _, lexicon_metrics = run_lexicon_models(
        lexicon_subset_df.reset_index(drop=True)
    )
    comparison_df = pd.concat(
        [
            ml_subset_summary_df,
            pd.DataFrame(
                [
                    metrics_row("VADER", lexicon_metrics["vader"]),
                    metrics_row("TextBlob", lexicon_metrics["textblob"]),
                    metrics_row("SentiWordNet", lexicon_metrics["sentiwordnet"]),
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values(by=["f1_weighted", "accuracy"], ascending=False)

    comparison_df.to_csv(TABLES_DIR / "phase2_model_comparison.csv", index=False)
    lexicon_test_results.to_csv(
        PREDICTIONS_DIR / "phase2_test_lexicon_predictions.csv", index=False
    )
    lexicon_subset_df.to_csv(
        PREDICTIONS_DIR / "phase2_lexicon_comparison_subset.csv", index=False
    )

    write_json(
        {
            "source_file": profile["source_file"],
            "raw_rows_after_empty_text_filter": profile[
                "raw_rows_after_empty_text_filter"
            ],
            "prepared_rows": profile["prepared_rows"],
            "duplicate_rows_removed": profile["duplicate_rows_removed"],
            "development_sample_rows": int(len(development_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "lexicon_comparison_subset_rows": int(len(lexicon_subset_df)),
            "random_state": DEFAULT_RANDOM_STATE,
            "label_mapping": profile["label_mapping"],
            "cross_validation_folds": 3,
            "ml_models": list(build_model_pipelines().keys()),
            "tuned_models": list(model_param_grids().keys()),
            "saved_prepared_paths": [str(path) for path in saved_paths],
            "saved_development_paths": [str(path) for path in development_paths],
            "comparison_note": "ML metrics in phase2_ml_model_summary.csv are on the full held-out development test split. phase2_model_comparison.csv evaluates ML and lexicon baselines on the same stratified lexicon comparison subset from the large dataset.",
        },
        METRICS_DIR / "phase2_split_summary.json",
    )

    return {
        "cv_summary": cv_summary_df,
        "tuning_summary": tuning_df,
        "ml_summary": ml_summary_df,
        "comparison": comparison_df,
        "ml_metrics": ml_metrics,
    }


if __name__ == "__main__":
    outputs = run_ml_pipeline()
    print("Phase 2 baseline training complete.")
    print(outputs["comparison"].to_string(index=False))
