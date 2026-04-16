import argparse
import json
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

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
from src.model_registry import (
    MODEL_SPECS,
    build_selected_pipelines,
    default_model_names,
    experimental_model_names,
    resolve_model_names,
)
from src.prepare_phase2 import prepare_phase2_artifacts
from src.utils import ensure_directories, write_json


def artifact_stem(model_name: str) -> str:
    return model_name.lower().replace(" ", "")


def load_comparison_subset(
    comparison_subset_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    subset_path = (
        Path(comparison_subset_path)
        if comparison_subset_path
        else PREDICTIONS_DIR / "phase2_lexicon_comparison_subset.csv"
    )
    metadata_path = METRICS_DIR / "phase2_comparison_subset_metadata.json"
    if not subset_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Phase 2 comparison subset artifacts were not found. Run `uv run python -m src.prepare_phase2` first."
        )
    subset_df = pd.read_csv(subset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return subset_df, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 sentiment models.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model CLI names to train, for example: logistic_regression svm complement_nb mlp",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Train default models plus experimental models.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation for faster runs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Held-out test fraction for the Phase 2 development sample.",
    )
    parser.add_argument(
        "--prepared-sample-path",
        type=str,
        default=None,
        help="Optional path to a prepared Phase 2 development sample CSV.",
    )
    parser.add_argument(
        "--comparison-subset-path",
        type=str,
        default=None,
        help="Optional path to a shared comparison subset CSV.",
    )
    return parser.parse_args()


def selected_cli_names(args: argparse.Namespace) -> list[str]:
    if args.models:
        return resolve_model_names(args.models)
    if args.include_experimental:
        return default_model_names() + experimental_model_names()
    return default_model_names()


def cross_validate_models(
    train_df: pd.DataFrame, selected_names: list[str]
) -> pd.DataFrame:
    rows = []
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE
    )
    train_text = cast(pd.Series, train_df["text"])
    train_label = cast(pd.Series, train_df["label"])
    for model_name, pipeline in build_selected_pipelines(selected_names).items():
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
    train_df: pd.DataFrame, test_df: pd.DataFrame, selected_names: list[str]
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

    for model_name, pipeline in build_selected_pipelines(selected_names).items():
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

        model_stem = artifact_stem(model_name)
        prediction_frame.to_csv(
            PREDICTIONS_DIR / f"phase2_{model_stem}_test_predictions.csv",
            index=False,
        )
        write_json(metrics, METRICS_DIR / f"phase2_{model_stem}_metrics.json")
        save_confusion_matrix(
            test_label,
            predictions,
            f"Phase 2 {model_name} Confusion Matrix",
            FIGURES_DIR / f"phase2_{model_stem}_confusion_matrix.png",
        )
        joblib.dump(fitted_pipeline, model_dir / f"phase2_{model_stem}.joblib")

        class_distribution, error_table = build_error_tables(
            prediction_frame, prediction_column
        )
        class_distribution.to_csv(
            TABLES_DIR / f"phase2_{model_stem}_class_distribution.csv", index=False
        )
        error_table.to_csv(
            TABLES_DIR / f"phase2_{model_stem}_error_analysis.csv", index=False
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


def save_ml_comparison_outputs(
    fitted_models: dict[str, Pipeline], subset_df: pd.DataFrame, subset_metadata: dict
) -> pd.DataFrame:
    rows = []
    subset_text = cast(pd.Series, subset_df["text"])
    subset_label = cast(pd.Series, subset_df["label"])

    for model_name, pipeline in fitted_models.items():
        predictions = pd.Series(pipeline.predict(subset_text), index=subset_df.index)
        metrics = compute_classification_metrics(subset_label, predictions)
        rows.append(metrics_row(model_name, metrics))

        model_stem = artifact_stem(model_name)
        prediction_frame = cast(
            pd.DataFrame,
            subset_df[["overall", "summary", "reviewText", "text", "label"]],
        ).copy()
        prediction_frame[f"{model_name}_pred"] = predictions.values
        prediction_frame.to_csv(
            PREDICTIONS_DIR / f"phase2_{model_stem}_comparison_predictions.csv",
            index=False,
        )
        write_json(
            {
                "model": model_name,
                "evaluation_scope": "shared_comparison_subset",
                "subset_metadata": subset_metadata,
                "metrics": metrics,
            },
            METRICS_DIR / f"phase2_{model_stem}_comparison_metrics.json",
        )

    return pd.DataFrame(rows).sort_values(
        by=["f1_weighted", "accuracy"], ascending=False
    )


def run_ml_pipeline(
    selected_names: list[str] | None = None,
    skip_cv: bool = False,
    test_size: float = 0.3,
    prepared_sample_path: str | None = None,
    comparison_subset_path: str | None = None,
) -> dict:
    resolved_names = resolve_model_names(selected_names)
    ensure_directories([FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR])
    outputs = prepare_phase2_artifacts()
    if prepared_sample_path:
        development_df = pd.read_csv(prepared_sample_path, low_memory=False)
    else:
        development_df = cast(pd.DataFrame, outputs["development_df"])
    profile = cast(dict, outputs["profile"])
    comparison_subset_df, comparison_subset_metadata = load_comparison_subset(
        comparison_subset_path
    )

    train_df, test_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            development_df,
            test_size=test_size,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=development_df["overall"],
        ),
    )

    if skip_cv:
        cv_summary_df = pd.DataFrame(
            columns=[
                "model",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_f1_weighted_mean",
                "cv_f1_weighted_std",
            ]
        )
    else:
        cv_summary_df = cross_validate_models(train_df, resolved_names)
        cv_summary_df.to_csv(
            TABLES_DIR / "phase2_cross_validation_summary.csv", index=False
        )

    ml_summary_df, ml_predictions_df, ml_metrics, fitted_models = evaluate_ml_models(
        train_df, test_df, resolved_names
    )
    ml_summary_df.to_csv(TABLES_DIR / "phase2_ml_model_summary.csv", index=False)
    ml_predictions_df.to_csv(
        PREDICTIONS_DIR / "phase2_ml_all_test_predictions.csv", index=False
    )

    comparison_df = save_ml_comparison_outputs(
        fitted_models,
        comparison_subset_df,
        comparison_subset_metadata,
    )
    lexicon_subset_size = int(len(comparison_subset_df))
    comparison_df.to_csv(TABLES_DIR / "phase2_ml_comparison_summary.csv", index=False)

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
            "lexicon_comparison_subset_rows": lexicon_subset_size,
            "comparison_subset_metadata": comparison_subset_metadata,
            "train_test_split": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
            "train_test_stratify_field": "overall",
            "skip_cv": skip_cv,
            "random_state": DEFAULT_RANDOM_STATE,
            "label_mapping": profile["label_mapping"],
            "cross_validation_folds": 0 if skip_cv else 3,
            "ml_models": [MODEL_SPECS[name].display_name for name in resolved_names],
            "selected_model_cli_names": resolved_names,
            "comparison_note": "ML metrics in phase2_ml_model_summary.csv are on the full held-out development test split. Per-model comparison metrics are saved from the same shared held-out test set used later by the lexicon testing script and the final compare_models aggregation step.",
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
    args = parse_args()
    model_names = selected_cli_names(args)
    outputs = run_ml_pipeline(
        model_names,
        skip_cv=args.skip_cv,
        test_size=args.test_size,
        prepared_sample_path=args.prepared_sample_path,
        comparison_subset_path=args.comparison_subset_path,
    )
    print("Phase 2 baseline training complete.")
    print(f"Trained models: {', '.join(model_names)}")
    print(outputs["comparison"].to_string(index=False))
