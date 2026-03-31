from pathlib import Path
import shutil
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import LABEL_ORDER, TABLES_DIR, FIGURES_DIR
from src.utils import ensure_directories


sns.set_theme(style="whitegrid")


def save_single_distribution(
    series: pd.Series,
    path: Path,
    title: str,
    xlabel: str,
    color: str,
    order: list | None = None,
) -> None:
    counts = series.value_counts().sort_index()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        [str(index) for index in counts.index],
        counts.values,
        color=color,
        edgecolor="black",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        axis.text(
            index,
            value + max(counts.values) * 0.01 if len(counts.values) else 0,
            str(int(value)),
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_review_length_distributions(
    df: pd.DataFrame, path: Path, title_prefix: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["review_char_count"], bins=50, color="teal", edgecolor="black")
    axes[0].set_title(f"{title_prefix} Review Length (Characters)")
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["review_word_count"], bins=50, color="salmon", edgecolor="black")
    axes[1].set_title(f"{title_prefix} Review Length (Words)")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reviews_per_entity(
    df: pd.DataFrame, column: str, path: Path, title: str, xlabel: str, ylabel: str
) -> None:
    counts = df.groupby(column).size()
    fig, axis = plt.subplots(figsize=(6, 4))
    _, _, patches = axis.hist(counts, bins=30, color="coral", edgecolor="black")
    axis.set_title(f"{title}\n(Number of {ylabel}: {len(counts)})")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    for patch in patches:
        height = patch.get_height()
        if height > 0:
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.2,
                str(int(height)),
                ha="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_missing_values(df: pd.DataFrame, path: Path) -> None:
    missing_share = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        list(missing_share.index.astype(str)),
        list(missing_share.values),
        color="gray",
        edgecolor="black",
    )
    axis.set_title("Missing Values by Column")
    axis.set_xlabel("Column")
    axis.set_ylabel("Missing Percentage")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reviewer_word_count_boxplot(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    q1 = float(df["review_word_count"].quantile(0.25))
    q3 = float(df["review_word_count"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[
        (df["review_word_count"] < lower_bound)
        | (df["review_word_count"] > upper_bound)
    ].copy()

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.boxplot(df["review_word_count"], vert=True)
    axis.set_title(f"Boxplot of Review Word Count\nOutlier reviews: {len(outliers)}")
    axis.set_ylabel("Word Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return pd.DataFrame(
        [
            {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_review_count": int(len(outliers)),
            }
        ]
    )


def save_reviewer_bias_analysis(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    reviewer_stats = (
        df.groupby("reviewerID")
        .agg(review_count=("asin", "count"), avg_rating=("overall", "mean"))
        .reset_index()
    )
    global_avg = float(df["overall"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        x="review_count",
        y="avg_rating",
        data=reviewer_stats,
        alpha=0.7,
        color="purple",
        ax=axes[0],
    )
    axes[0].set_title("Reviewer Bias: Review Count vs. Avg Rating")
    axes[0].set_xlabel("Number of Reviews by User")
    axes[0].set_ylabel("Average Rating Given")
    axes[0].axhline(
        global_avg, color="red", linestyle="--", label=f"Global Avg ({global_avg:.2f})"
    )
    axes[0].legend()

    sns.histplot(
        reviewer_stats["avg_rating"], bins=10, kde=True, color="teal", ax=axes[1]
    )
    axes[1].set_title(
        f"Distribution of Average Ratings per Reviewer\n(Number of Unique Reviewers: {len(reviewer_stats)})"
    )
    axes[1].set_xlabel("Average Rating")
    axes[1].set_ylabel("Number of Reviewers")
    for patch in axes[1].patches:
        height = patch.get_height()
        if height > 0:
            axes[1].text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.5,
                str(int(height)),
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return reviewer_stats


def save_reviewer_rating_map(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    high_freq_df = df.groupby("reviewerID").filter(lambda rows: len(rows) > 200)
    if high_freq_df.empty:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "No reviewers with >200 reviews", ha="center", va="center")
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return pd.DataFrame()

    rating_dist = pd.crosstab(high_freq_df["reviewerID"], high_freq_df["overall"])
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        rating_dist,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Number of Reviews"},
        ax=axis,
    )
    axis.set_title("Rating Distribution per High-Frequency Reviewer (>200 Reviews)")
    axis.set_xlabel("Star Rating (Overall)")
    axis.set_ylabel("Reviewer ID")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rating_dist.reset_index()


def comparison_counts(
    before: pd.Series, after: pd.Series, categories: list
) -> pd.DataFrame:
    before_counts = before.value_counts().reindex(categories, fill_value=0)
    after_counts = after.value_counts().reindex(categories, fill_value=0)
    comparison = pd.DataFrame(
        {
            "category": categories,
            "before_duplicate_removal": before_counts.values,
            "after_duplicate_removal": after_counts.values,
        }
    )
    comparison["removed_by_duplicate_step"] = (
        comparison["before_duplicate_removal"] - comparison["after_duplicate_removal"]
    )
    comparison["removed_share_pct"] = np.where(
        comparison["before_duplicate_removal"] == 0,
        0.0,
        comparison["removed_by_duplicate_step"]
        / comparison["before_duplicate_removal"]
        * 100,
    )
    return comparison


def save_before_after_bar_chart(
    comparison_df: pd.DataFrame,
    category_column: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    chart_df = comparison_df.set_index(category_column)[
        ["before_duplicate_removal", "after_duplicate_removal"]
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    chart_df.plot(kind="bar", ax=axis, color=["cornflowerblue", "darkorange"])
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.legend(["Before duplicate removal", "After duplicate removal"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_removed_share_chart(
    comparison_df: pd.DataFrame,
    category_column: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        comparison_df[category_column].astype(str),
        comparison_df["removed_share_pct"],
        color="indianred",
        edgecolor="black",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Removed share (%)")
    for index, value in enumerate(comparison_df["removed_share_pct"]):
        axis.text(index, float(value) + 0.5, f"{float(value):.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_phase1_overview_table(
    raw_df: pd.DataFrame, prepared_df: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "rows_before_duplicate_removal", "value": len(raw_df)},
            {"metric": "rows_after_duplicate_removal", "value": len(prepared_df)},
            {"metric": "duplicates_removed", "value": len(raw_df) - len(prepared_df)},
            {
                "metric": "duplicate_removal_share_pct",
                "value": (len(raw_df) - len(prepared_df)) / len(raw_df) * 100,
            },
            {
                "metric": "unique_products_after_duplicate_removal",
                "value": prepared_df["asin"].nunique()
                if "asin" in prepared_df.columns
                else 0,
            },
            {
                "metric": "unique_reviewers_after_duplicate_removal",
                "value": prepared_df["reviewerID"].nunique()
                if "reviewerID" in prepared_df.columns
                else 0,
            },
            {
                "metric": "mean_rating_before_duplicate_removal",
                "value": raw_df["overall"].mean(),
            },
            {
                "metric": "mean_rating_after_duplicate_removal",
                "value": prepared_df["overall"].mean(),
            },
            {
                "metric": "mean_review_words_before_duplicate_removal",
                "value": raw_df["review_word_count"].mean(),
            },
            {
                "metric": "mean_review_words_after_duplicate_removal",
                "value": prepared_df["review_word_count"].mean(),
            },
            {
                "metric": "verified_reviews_after_duplicate_removal",
                "value": int(prepared_df["verified"].sum())
                if "verified" in prepared_df.columns
                else 0,
            },
            {
                "metric": "unverified_reviews_after_duplicate_removal",
                "value": int((~prepared_df["verified"]).sum())
                if "verified" in prepared_df.columns
                else 0,
            },
        ]
    )


def save_dataset_exploration_outputs(
    raw_df: pd.DataFrame, prepared_df: pd.DataFrame, prefix: str
) -> None:
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    save_single_distribution(
        cast(pd.Series, raw_df["overall"]),
        FIGURES_DIR / f"{prefix}_raw_rating_distribution.png",
        "Rating Distribution Before Duplicate Removal",
        "Star Rating",
        "steelblue",
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    save_single_distribution(
        cast(pd.Series, prepared_df["overall"]),
        FIGURES_DIR / f"{prefix}_prepared_rating_distribution.png",
        "Rating Distribution After Duplicate Removal",
        "Star Rating",
        "darkorange",
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    save_single_distribution(
        cast(pd.Series, raw_df["label"]),
        FIGURES_DIR / f"{prefix}_raw_label_distribution.png",
        "Label Distribution Before Duplicate Removal",
        "Derived Label",
        "seagreen",
        LABEL_ORDER,
    )
    save_single_distribution(
        cast(pd.Series, prepared_df["label"]),
        FIGURES_DIR / f"{prefix}_prepared_label_distribution.png",
        "Label Distribution After Duplicate Removal",
        "Derived Label",
        "firebrick",
        LABEL_ORDER,
    )
    save_review_length_distributions(
        raw_df,
        FIGURES_DIR / f"{prefix}_raw_review_length_distributions.png",
        "Before Duplicate Removal",
    )
    save_review_length_distributions(
        prepared_df,
        FIGURES_DIR / f"{prefix}_prepared_review_length_distributions.png",
        "After Duplicate Removal",
    )

    reviewer_word_count_summary = save_reviewer_word_count_boxplot(
        raw_df, FIGURES_DIR / f"{prefix}_reviewer_word_count.png"
    )
    reviewer_stats = save_reviewer_bias_analysis(
        raw_df, FIGURES_DIR / f"{prefix}_reviewer_bias_analysis.png"
    )
    reviewer_rating_map = save_reviewer_rating_map(
        raw_df, FIGURES_DIR / f"{prefix}_reviewer_rating_map.png"
    )
    save_missing_values(raw_df, FIGURES_DIR / f"{prefix}_missing_values.png")
    save_reviews_per_entity(
        prepared_df,
        "asin",
        FIGURES_DIR / f"{prefix}_reviews_per_product.png",
        "Distribution of Reviews Per Product After Duplicate Removal",
        "Reviews per product",
        "Products",
    )
    save_reviews_per_entity(
        prepared_df,
        "reviewerID",
        FIGURES_DIR / f"{prefix}_reviews_per_reviewer.png",
        "Distribution of Reviews Per Reviewer After Duplicate Removal",
        "Reviews per reviewer",
        "Reviewers",
    )

    rating_comparison = comparison_counts(
        cast(pd.Series, raw_df["overall"]),
        cast(pd.Series, prepared_df["overall"]),
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    label_comparison = comparison_counts(
        cast(pd.Series, raw_df["label"]),
        cast(pd.Series, prepared_df["label"]),
        LABEL_ORDER,
    )
    save_before_after_bar_chart(
        rating_comparison,
        "category",
        FIGURES_DIR / f"{prefix}_rating_distribution_before_after.png",
        "Ratings Before vs After Duplicate Removal",
        "Star Rating",
    )
    save_before_after_bar_chart(
        label_comparison,
        "category",
        FIGURES_DIR / f"{prefix}_label_distribution_before_after.png",
        "Labels Before vs After Duplicate Removal",
        "Derived Label",
    )
    save_removed_share_chart(
        rating_comparison,
        "category",
        FIGURES_DIR / f"{prefix}_rating_duplicate_impact.png",
        "Duplicate Removal Impact by Rating",
        "Star Rating",
    )
    save_removed_share_chart(
        label_comparison,
        "category",
        FIGURES_DIR / f"{prefix}_label_duplicate_impact.png",
        "Duplicate Removal Impact by Label",
        "Derived Label",
    )

    build_phase1_overview_table(raw_df, prepared_df).to_csv(
        TABLES_DIR / f"{prefix}_dataset_overview.csv", index=False
    )
    rating_comparison.to_csv(
        TABLES_DIR / f"{prefix}_rating_duplicate_impact.csv", index=False
    )
    label_comparison.to_csv(
        TABLES_DIR / f"{prefix}_label_duplicate_impact.csv", index=False
    )
    reviewer_word_count_summary.to_csv(
        TABLES_DIR / f"{prefix}_reviewer_word_count_summary.csv", index=False
    )
    reviewer_stats.to_csv(TABLES_DIR / f"{prefix}_reviewer_bias_stats.csv", index=False)
    reviewer_rating_map.to_csv(
        TABLES_DIR / f"{prefix}_reviewer_rating_map.csv", index=False
    )


def save_phase1_exploration_outputs(
    raw_df: pd.DataFrame, prepared_df: pd.DataFrame
) -> None:
    save_dataset_exploration_outputs(raw_df, prepared_df, prefix="phase1")
    shutil.copyfile(
        FIGURES_DIR / "phase1_reviewer_word_count.png",
        FIGURES_DIR / "reviewer_word_count.png",
    )
    shutil.copyfile(
        FIGURES_DIR / "phase1_reviewer_bias_analysis.png",
        FIGURES_DIR / "reviewer_bias_analysis.png",
    )
    shutil.copyfile(
        FIGURES_DIR / "phase1_reviewer_rating_map.png",
        FIGURES_DIR / "reviewer_rating_map.png",
    )
