from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    DEFAULT_RANDOM_STATE,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
)
from src.evaluate import (
    compute_classification_metrics,
    metrics_row,
    save_confusion_matrix,
)
from src.features import build_tfidf_vectorizer
from src.lexicon_baselines import run_lexicon_models
from src.prepare_phase2 import build_lexicon_comparison_subset, prepare_phase2_artifacts
from src.utils import ensure_directories, write_json


def build_model_pipelines() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer().set_params(min_df=1)),
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
                        C=0.5,
                        class_weight="balanced",
                        random_state=DEFAULT_RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MultinomialNB": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer().set_params(min_df=2)),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        ),
    }


def cross_validate_models(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE
    )
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
    return class_distribution, errors[
        ["label", prediction_column, "overall", "summary", "text_preview"]
    ].reset_index(drop=True)


def evaluate_ml_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, Pipeline]]:
    model_dir = Path("outputs") / "models"
    ensure_directories([model_dir])
    prediction_frames = []
    metrics_by_model = {}
    fitted_models: dict[str, Pipeline] = {}
    rows = []
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
        rows.append(metrics_row(model_name, metrics))

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

    return (
        pd.DataFrame(rows).sort_values(by=["f1_weighted", "accuracy"], ascending=False),
        pd.concat(prediction_frames, ignore_index=True),
        metrics_by_model,
        fitted_models,
    )


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
    outputs = prepare_phase2_artifacts()
    development_df = cast(pd.DataFrame, outputs["development_df"])
    profile = cast(dict, outputs["profile"])

    train_df, test_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            development_df,
            test_size=0.3,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=development_df["overall"],
        ),
    )

    cv_summary_df = cross_validate_models(train_df)
    cv_summary_df.to_csv(
        TABLES_DIR / "phase2_cross_validation_summary.csv", index=False
    )

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
            "train_test_split": "70/30",
            "train_test_stratify_field": "overall",
            "random_state": DEFAULT_RANDOM_STATE,
            "label_mapping": profile["label_mapping"],
            "cross_validation_folds": 3,
            "ml_models": list(build_model_pipelines().keys()),
            "comparison_note": "ML metrics in phase2_ml_model_summary.csv are on the full held-out development test split. phase2_model_comparison.csv evaluates ML and lexicon baselines on the same stratified lexicon comparison subset from the large dataset.",
        },
        METRICS_DIR / "phase2_split_summary.json",
    )

    return {
        "cv_summary": cv_summary_df,
        "ml_summary": ml_summary_df,
        "comparison": comparison_df,
        "ml_metrics": ml_metrics,
    }


if __name__ == "__main__":
    outputs = run_ml_pipeline()
    print("Phase 2 baseline training complete.")
    print(outputs["comparison"].to_string(index=False))
