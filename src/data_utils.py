import re
from pathlib import Path

import pandas as pd

from src.config import (
    LARGE_RAW_REVIEW_FILE,
    PROCESSED_DIR,
    SECTION16_MIN_WORD_COUNT,
    SECTION16_SUMMARY_COUNT,
    SECTION17_QUESTION_PHRASES,
)
from src.prepare_phase2 import prepare_phase2_artifacts


PHASE2_PROCESSED_PATH = PROCESSED_DIR / "amazon_appliances_large_reviews_labeled.csv"


def load_phase2_reviews() -> pd.DataFrame:
    """Load the prepared large Phase 2 dataset, preparing it if needed."""
    if not PHASE2_PROCESSED_PATH.exists():
        prepare_phase2_artifacts()
    return pd.read_csv(PHASE2_PROCESSED_PATH, low_memory=False)


def load_phase2_comparison_subset() -> pd.DataFrame:
    subset_path = (
        Path("outputs") / "predictions" / "phase2_lexicon_comparison_subset.csv"
    )
    if not subset_path.exists():
        prepare_phase2_artifacts()
    return pd.read_csv(subset_path, low_memory=False)


def word_count(text: str) -> int:
    return len(str(text).split())


def text_excerpt(text: str, max_chars: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def select_long_reviews(
    df: pd.DataFrame,
    count: int = SECTION16_SUMMARY_COUNT,
    min_word_count: int = SECTION16_MIN_WORD_COUNT,
) -> pd.DataFrame:
    """Select exactly `count` long reviews deterministically."""
    long_reviews = df.loc[df["review_word_count"] > min_word_count].copy()
    ordered = long_reviews.sort_values(
        by=["review_word_count", "unixReviewTime", "reviewerID", "asin"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return ordered.head(count).reset_index(names="dataset_index")


def question_like_score(text: str) -> int:
    lower_text = str(text).strip().lower()
    score = 0
    if "?" in lower_text:
        score += 3
    if lower_text.startswith("what can i say"):
        score -= 2
    for phrase in SECTION17_QUESTION_PHRASES:
        if phrase in lower_text:
            score += 2
    if "do i need" in lower_text:
        score += 2
    if "can it" in lower_text or "can this" in lower_text:
        score += 2
    if lower_text.startswith(("can ", "does ", "is ", "will ", "how ", "what ")):
        score += 1
    return score


def select_question_like_review(df: pd.DataFrame) -> pd.Series:
    """Choose one question-like review deterministically."""
    working = df.copy()
    working["question_score"] = working["text"].map(question_like_score)
    candidates = working.loc[working["question_score"] > 0].copy()
    candidates = candidates.loc[candidates["review_word_count"].between(5, 180)].copy()
    ordered = candidates.sort_values(
        by=["question_score", "review_word_count", "unixReviewTime", "reviewerID"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def dataset_schema_summary() -> dict:
    return {
        "source_file": str(LARGE_RAW_REVIEW_FILE),
        "prepared_path": str(PHASE2_PROCESSED_PATH),
        "preferred_text_field": "text",
        "text_components": ["summary", "reviewText"],
        "rating_field": "overall",
        "useful_fields": [
            "summary",
            "reviewText",
            "text",
            "overall",
            "review_word_count",
            "verified",
            "vote",
            "asin",
            "reviewerID",
        ],
    }
