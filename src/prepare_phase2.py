import argparse
from pathlib import Path
from typing import cast

import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_PHASE2_DEV_SAMPLE_SIZE,
    DEFAULT_PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE,
    DEFAULT_RANDOM_STATE,
    LARGE_RAW_REVIEW_FILE,
    METRICS_DIR,
    PREDICTIONS_DIR,
)
from src.data_prep import (
    load_amazon_reviews,
    prepare_dataset,
    resolve_large_raw_data_path,
    save_prepared_dataset,
)
from src.phase1_exploration import save_dataset_exploration_outputs
from src.utils import ensure_directories, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 2 large-dataset artifacts."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_PHASE2_DEV_SAMPLE_SIZE,
        help="Development sample size for Phase 2.",
    )
    parser.add_argument(
        "--comparison-size",
        type=int,
        default=DEFAULT_PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE,
        help="Shared comparison subset size for lexicon-vs-ML evaluation.",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Override path for the large raw dataset file.",
    )
    parser.add_argument(
        "--skip-exploration",
        action="store_true",
        help="Skip generation of exploration figures and tables.",
    )
    return parser.parse_args()


def build_subset_metadata(df: pd.DataFrame, subset_name: str) -> dict:
    return {
        "subset_name": subset_name,
        "rows": int(len(df)),
        "overall_distribution": {
            str(key): int(value)
            for key, value in df["overall"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "label_distribution": {
            key: int(value)
            for key, value in df["label"].value_counts().to_dict().items()
        },
        "text_hash_sum": int(hash_pandas_object(df["text"], index=False).sum()),
    }


def resolve_phase2_raw_data_path(raw_data_path: str | None = None) -> Path:
    if raw_data_path:
        return Path(raw_data_path)
    return resolve_large_raw_data_path()


def build_phase2_dataset(
    raw_data_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, tuple[Path, Path]]:
    source_path = resolve_phase2_raw_data_path(raw_data_path)
    raw_loaded_df = load_amazon_reviews(source_path)
    raw_prepared_df = prepare_dataset(raw_loaded_df, remove_exact_duplicates=False)
    prepared_df = prepare_dataset(raw_loaded_df, remove_exact_duplicates=True)
    profile = {
        "source_file": str(source_path),
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


def build_development_sample(
    prepared_df: pd.DataFrame, sample_size: int
) -> pd.DataFrame:
    if len(prepared_df) <= sample_size:
        return prepared_df.reset_index(drop=True)
    development_df, _ = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            prepared_df,
            train_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=prepared_df["overall"],
        ),
    )
    return development_df.reset_index(drop=True)


def build_lexicon_comparison_subset(
    test_df: pd.DataFrame, comparison_size: int
) -> pd.DataFrame:
    if len(test_df) <= comparison_size:
        return test_df.reset_index(drop=True)
    lexicon_subset_df, _ = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            test_df,
            train_size=comparison_size,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=test_df["overall"],
        ),
    )
    return lexicon_subset_df.reset_index(drop=True)


def prepare_phase2_artifacts(
    sample_size: int = DEFAULT_PHASE2_DEV_SAMPLE_SIZE,
    comparison_size: int = DEFAULT_PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE,
    raw_data_path: str | None = None,
    skip_exploration: bool = False,
) -> dict:
    ensure_directories([METRICS_DIR, PREDICTIONS_DIR])
    raw_prepared_df, prepared_df, profile, saved_paths = build_phase2_dataset(
        raw_data_path
    )
    if not skip_exploration:
        save_dataset_exploration_outputs(raw_prepared_df, prepared_df, prefix="phase2")

    development_df = build_development_sample(prepared_df, sample_size)
    development_paths = save_prepared_dataset(
        development_df,
        dataset_name="amazon_appliances_large_phase2_development_sample",
    )
    _, test_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            development_df,
            test_size=0.3,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=development_df["overall"],
        ),
    )
    comparison_subset_df = build_lexicon_comparison_subset(test_df, comparison_size)
    comparison_subset_path = PREDICTIONS_DIR / "phase2_lexicon_comparison_subset.csv"
    comparison_subset_df.to_csv(comparison_subset_path, index=False)
    comparison_subset_metadata = {
        "source_file": profile["source_file"],
        "train_test_split": "70/30",
        "train_test_stratify_field": "overall",
        "random_state": DEFAULT_RANDOM_STATE,
        "development_sample_size": int(sample_size),
        "comparison_subset_size": int(comparison_size),
        "subset": build_subset_metadata(
            comparison_subset_df, "phase2_lexicon_comparison_subset"
        ),
        "subset_path": str(comparison_subset_path),
    }
    write_json(
        comparison_subset_metadata,
        METRICS_DIR / "phase2_comparison_subset_metadata.json",
    )

    write_json(
        {
            **profile,
            "development_sample_rows": int(len(development_df)),
            "development_sample_size_requested": int(sample_size),
            "comparison_subset_size_requested": int(comparison_size),
            "saved_prepared_paths": [str(path) for path in saved_paths],
            "saved_development_paths": [str(path) for path in development_paths],
            "saved_comparison_subset_path": str(comparison_subset_path),
            "skip_exploration": skip_exploration,
        },
        METRICS_DIR / "phase2_prepare_summary.json",
    )

    return {
        "raw_prepared_df": raw_prepared_df,
        "prepared_df": prepared_df,
        "development_df": development_df,
        "profile": profile,
        "saved_paths": saved_paths,
        "development_paths": development_paths,
        "comparison_subset_df": comparison_subset_df,
        "comparison_subset_path": comparison_subset_path,
        "comparison_subset_metadata": comparison_subset_metadata,
    }


if __name__ == "__main__":
    args = parse_args()
    outputs = prepare_phase2_artifacts(
        sample_size=args.sample_size,
        comparison_size=args.comparison_size,
        raw_data_path=args.raw_data_path,
        skip_exploration=args.skip_exploration,
    )
    print("Phase 2 data preparation complete.")
    print(f"Prepared rows: {len(outputs['prepared_df'])}")
    print(f"Development sample rows: {len(outputs['development_df'])}")
