import gzip
import json
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_RANDOM_STATE,
    INTERIM_DIR,
    LARGE_RAW_REVIEW_FILE,
    PROCESSED_DIR,
    RAW_REVIEW_FILES,
    RATING_LABEL_MAPPING,
    SMALL_RAW_REVIEW_FILE,
)
from src.utils import ensure_directories


def resolve_raw_data_path() -> Path:
    for path in RAW_REVIEW_FILES:
        if path.exists():
            return path
    raise FileNotFoundError("No expected raw review file was found in data/raw/.")


def resolve_small_raw_data_path() -> Path:
    if SMALL_RAW_REVIEW_FILE.exists():
        return SMALL_RAW_REVIEW_FILE
    raise FileNotFoundError(
        "The small review dataset Appliances_5.json.gz was not found."
    )


def resolve_large_raw_data_path() -> Path:
    if LARGE_RAW_REVIEW_FILE.exists():
        return LARGE_RAW_REVIEW_FILE
    raise FileNotFoundError(
        "The large review dataset Appliances.json.gz was not found."
    )


def load_amazon_reviews(path: Path | None = None) -> pd.DataFrame:
    source_path = path or resolve_raw_data_path()
    records = []
    with gzip.open(source_path, "rt", encoding="utf-8") as file_handle:
        for line in file_handle:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def label_from_rating(rating: float) -> str:
    if pd.isna(rating):
        raise ValueError("Rating cannot be missing when deriving labels.")
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def combine_review_text(summary: str | float, review_text: str | float) -> str:
    left = "" if pd.isna(summary) else str(summary).strip()
    right = "" if pd.isna(review_text) else str(review_text).strip()
    pieces = [piece for piece in (left, right) if piece]
    return ". ".join(pieces)


def prepare_dataset(
    df: pd.DataFrame, remove_exact_duplicates: bool = True
) -> pd.DataFrame:
    working = df.copy()
    if "reviewText" not in working.columns or "overall" not in working.columns:
        raise ValueError("The dataset must include 'reviewText' and 'overall' columns.")

    working["reviewText"] = working["reviewText"].fillna("").astype(str)
    if "summary" not in working.columns:
        working["summary"] = ""
    working["summary"] = working["summary"].fillna("").astype(str)

    working = working.loc[working["reviewText"].str.strip() != ""].copy()

    if remove_exact_duplicates:
        subset = [
            column
            for column in ["reviewerID", "asin", "reviewText"]
            if column in working.columns
        ]
        if subset:
            working = working.drop_duplicates(subset=subset).reset_index(drop=True)

    working["text"] = working.apply(
        lambda row: combine_review_text(
            row.get("summary", ""), row.get("reviewText", "")
        ),
        axis=1,
    )
    working["label"] = working["overall"].apply(label_from_rating)
    working["review_char_count"] = working["reviewText"].str.len()
    working["review_word_count"] = working["reviewText"].str.split().str.len()
    return working


def build_data_profile(df: pd.DataFrame, source_path: Path) -> dict:
    return {
        "source_file": str(source_path.relative_to(source_path.parents[2])),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_mapping": RATING_LABEL_MAPPING,
        "rating_distribution": {
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
        "duplicate_policy": "Drop exact duplicates by reviewerID, asin, and reviewText.",
        "random_state": DEFAULT_RANDOM_STATE,
    }


def save_prepared_dataset(
    df: pd.DataFrame, dataset_name: str = "amazon_appliances_reviews"
) -> tuple[Path, Path]:
    ensure_directories([INTERIM_DIR, PROCESSED_DIR])
    interim_path = INTERIM_DIR / f"{dataset_name}_prepared.csv"
    processed_path = PROCESSED_DIR / f"{dataset_name}_labeled.csv"
    df.to_csv(interim_path, index=False)
    df.to_csv(processed_path, index=False)
    return interim_path, processed_path


def prepare_and_save_dataset(
    path: Path | None = None,
    dataset_name: str = "amazon_appliances_reviews",
) -> tuple[pd.DataFrame, dict, tuple[Path, Path]]:
    source_path = path or resolve_raw_data_path()
    raw_df = load_amazon_reviews(source_path)
    prepared_df = prepare_dataset(raw_df)
    profile = build_data_profile(prepared_df, source_path)
    saved_paths = save_prepared_dataset(prepared_df, dataset_name=dataset_name)
    return prepared_df, profile, saved_paths


if __name__ == "__main__":
    prepared_df, profile, saved_paths = prepare_and_save_dataset()
    print(f"Prepared dataset rows: {len(prepared_df)}")
    print(f"Saved interim dataset: {saved_paths[0]}")
    print(f"Saved processed dataset: {saved_paths[1]}")
    print(f"Label mapping: {profile['label_mapping']}")
