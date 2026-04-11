import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import cast

from src.config import MODELS_DIR, REPORTS_DIR, TABLES_DIR
from src.data_utils import load_phase2_reviews, text_excerpt
from src.utils import ensure_directories, markdown_table_from_rows, write_markdown


ALPHAS = [0.0, 0.9, 0.8, 0.7, 0.6]
LABEL_TO_RATING = {"Negative": 1.5, "Neutral": 3.0, "Positive": 4.5}


def load_best_sentiment_model():
    """Load the strongest existing repo model that supports text inference."""
    candidate_paths = [
        MODELS_DIR / "phase2_mlp.joblib",
        MODELS_DIR / "phase2_multinomialnb.joblib",
        MODELS_DIR / "phase2_linearsvc.joblib",
        MODELS_DIR / "phase2_logisticregression.joblib",
    ]
    for model_path in candidate_paths:
        if model_path.exists():
            return joblib.load(model_path), model_path.name
    raise FileNotFoundError(
        "No trained Phase 2 sentiment model artifact was found under outputs/models/."
    )


def infer_numeric_rating(model, texts: pd.Series) -> tuple[np.ndarray, pd.Series]:
    labels = pd.Series(model.predict(texts), index=texts.index)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        class_names = list(model.classes_)
        inferred = np.zeros(len(texts), dtype=float)
        for class_index, class_name in enumerate(class_names):
            inferred += probabilities[:, class_index] * LABEL_TO_RATING[class_name]
        return inferred, labels
    inferred = labels.map(LABEL_TO_RATING).to_numpy(dtype=float)
    return inferred, labels


def evaluate_alpha_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in ALPHAS:
        enhanced = alpha * df["overall"] + (1 - alpha) * df["inferred_rating"]
        rows.append(
            {
                "alpha": alpha,
                "mae": float(mean_absolute_error(df["overall"], enhanced)),
                "rmse": float(np.sqrt(mean_squared_error(df["overall"], enhanced))),
                "enhanced_mean": float(enhanced.mean()),
                "enhanced_std": float(enhanced.std()),
            }
        )
        df[f"enhanced_rating_alpha_{alpha:.1f}"] = enhanced
    return pd.DataFrame(rows)


def build_example_rows(df: pd.DataFrame, best_alpha: float) -> pd.DataFrame:
    working = df.copy()
    working["rating_gap"] = (working["overall"] - working["inferred_rating"]).abs()
    selected = working.sort_values(
        by=["rating_gap", "review_word_count"],
        ascending=[False, False],
        kind="mergesort",
    ).head(5)
    return pd.DataFrame(
        {
            "dataset_index": selected.index,
            "text_excerpt": selected["text"].map(text_excerpt),
            "original_rating": selected["overall"],
            "inferred_label": selected["inferred_label"],
            "inferred_rating": selected["inferred_rating"].round(3),
            "enhanced_rating": selected[
                f"enhanced_rating_alpha_{best_alpha:.1f}"
            ].round(3),
        }
    )


def rounded_distribution(values: pd.Series | np.ndarray) -> dict[str, int]:
    rounded = pd.Series(values).round().clip(lower=1, upper=5)
    return {
        str(key): int(value)
        for key, value in rounded.value_counts().sort_index().to_dict().items()
    }


def distribution_markdown(
    original: pd.Series, inferred: np.ndarray, enhanced: pd.Series
) -> str:
    original_dist = rounded_distribution(original)
    inferred_dist = rounded_distribution(inferred)
    enhanced_dist = rounded_distribution(enhanced)
    lines = [
        "| Rating | Original | Inferred | Enhanced |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rating in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
        lines.append(
            f"| {rating} | {original_dist.get(rating, 0)} | {inferred_dist.get(rating, 0)} | {enhanced_dist.get(rating, 0)} |"
        )
    return "\n".join(lines)


def build_report(
    metrics_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    model_name: str,
    df: pd.DataFrame,
) -> str:
    best_row = metrics_df.sort_values(by=["rmse", "mae"], ascending=True).iloc[0]
    best_alpha = float(best_row["alpha"])
    best_enhanced = df[f"enhanced_rating_alpha_{best_alpha:.1f}"]
    metrics_markdown = markdown_table_from_rows(
        list(metrics_df.columns), metrics_df.round(4).values.tolist()
    )
    examples_markdown = markdown_table_from_rows(
        list(examples_df.columns), examples_df.values.tolist()
    )
    distribution_table = distribution_markdown(
        cast(pd.Series, df["overall"]),
        cast(np.ndarray, df["inferred_rating"].to_numpy()),
        cast(pd.Series, best_enhanced),
    )
    return f"""# Section 15 - Rating Enhancement

## Method Name

Overall-opinion-enhanced rating blend.

## Why This Method Was Chosen

This project brief explicitly asks for a review-based enhancement of rating values. A simplified overall-opinion method is the best fit for this repo because it reuses the strongest existing sentiment model, works directly with the available review text, and avoids the extra complexity of topic, helpfulness, or aspect pipelines that are not already implemented here.

## Paper-Inspired Approach

The method estimates an inferred opinion from the review text, maps that opinion to the same 1-5 rating scale as the original Amazon rating, then blends the two values with a tunable alpha. This follows the brief's simplified interpretation of the paper idea that overall opinions from reviews can refine explicit ratings.

## ASCII Workflow Diagram

```text
review text + original rating
        |
        v
 preprocess / select usable text
        |
        v
 infer overall opinion with trained sentiment model
        |
        v
 map opinion to inferred 1-5 rating
        |
        v
 blend with original rating using alpha
        |
        v
 evaluate MAE and RMSE across alpha values
```

## Pseudo-code

```text
FOR each review:
    preprocess text
    infer overall opinion
    map opinion to inferred rating
    combine inferred rating with original rating using alpha
END FOR
evaluate multiple alpha values
select and discuss the best setting
```

## Assumptions

- The prepared large Phase 2 dataset is the correct source for this section.
- The merged `text` field, built from `summary` and `reviewText`, is the best available field for overall-opinion inference.
- The best existing model artifact available in the repo is `{model_name}`.
- The original `overall` rating is treated as the evaluation reference because no external gold enhanced-rating target exists.
- Sentiment labels are mapped to the 1-5 scale with anchors: Negative -> 1.5, Neutral -> 3.0, Positive -> 4.5.
- When the model exposes class probabilities, the inferred numeric rating is the probability-weighted expected rating on this 1-5 anchor scale.

## Preprocessing Assumptions

- Rows with missing or empty review text are already removed in the prepared dataset.
- Exact duplicate review rows are already removed by the existing Phase 2 preprocessing pipeline.
- The merged `text` field is reused as-is to keep this section consistent with the sentiment models already trained in the repo.

## Usable Row Count

- {len(df)} reviews

## Sentiment-to-Rating Mapping

- Negative -> 1.5
- Neutral -> 3.0
- Positive -> 4.5

## Results

### Alpha Evaluation

{metrics_markdown}

### Distribution Comparison Using Best Alpha ({best_alpha:.1f})

{distribution_table}

### Example Rows

{examples_markdown}

## Conclusion

The best alpha on this evaluation setup is {best_alpha:.1f}, which is expected because the original rating is used as the reference target. Even so, the experiment demonstrates a reproducible way to incorporate overall opinion inferred from review text into a blended rating value, which satisfies the project brief's rating-enhancement requirement.
"""


def main() -> None:
    ensure_directories([TABLES_DIR, REPORTS_DIR])
    df = load_phase2_reviews().copy()
    model, model_name = load_best_sentiment_model()
    inferred_rating, inferred_label = infer_numeric_rating(
        model, cast(pd.Series, df["text"])
    )
    df["inferred_rating"] = inferred_rating
    df["inferred_label"] = inferred_label

    metrics_df = evaluate_alpha_values(df)
    best_alpha = float(metrics_df.sort_values(by=["rmse", "mae"]).iloc[0]["alpha"])
    examples_df = build_example_rows(df, best_alpha)

    metrics_df.to_csv(TABLES_DIR / "rating_enhancement_metrics.csv", index=False)
    examples_df.to_csv(TABLES_DIR / "rating_enhancement_examples.csv", index=False)
    report_text = build_report(metrics_df, examples_df, model_name, df)
    write_markdown(report_text, REPORTS_DIR / "section15_rating_enhancement.md")

    print("Section 15 rating enhancement complete.")
    print(f"Rows used: {len(df)}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
