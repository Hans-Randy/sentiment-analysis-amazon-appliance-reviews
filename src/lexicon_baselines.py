import re
from pathlib import Path
from typing import Any

import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import (
    FIGURES_DIR,
    LABEL_ORDER,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
)
from src.data_prep import (
    load_amazon_reviews,
    prepare_and_save_dataset,
    prepare_dataset,
    resolve_small_raw_data_path,
)
from src.evaluate import (
    compute_classification_metrics,
    metrics_row,
    save_confusion_matrix,
)
from src.phase1_exploration import save_phase1_exploration_outputs
from src.utils import ensure_directories, write_json


def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/sentiwordnet", "sentiwordnet"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


def text_for_vader(text: str) -> str:
    return text.strip()


def text_for_textblob(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_for_sentiwordnet(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()


def vader_predict(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[float, str]:
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return score, "Positive"
    if score <= -0.05:
        return score, "Negative"
    return score, "Neutral"


def textblob_predict(text: str) -> tuple[float, str]:
    sentiment: Any = TextBlob(text).sentiment
    score = float(sentiment.polarity)
    if score >= 0.1:
        return score, "Positive"
    if score <= -0.1:
        return score, "Negative"
    return score, "Neutral"


def penn_to_wordnet(tag: str):
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return None


def sentiwordnet_predict(text: str) -> tuple[float, str]:
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    sentiment_score = 0.0

    for token, tag in tagged_tokens:
        wn_tag = penn_to_wordnet(tag)
        if wn_tag is None:
            continue
        synsets = list(swn.senti_synsets(token, wn_tag))
        if not synsets:
            continue

        pos_all = sum(synset.pos_score() for synset in synsets) / len(synsets)
        neg_all = sum(synset.neg_score() for synset in synsets) / len(synsets)
        sentiment_score += pos_all - neg_all

    if sentiment_score > 0.05:
        return sentiment_score, "Positive"
    if sentiment_score < -0.05:
        return sentiment_score, "Negative"
    return sentiment_score, "Neutral"


def run_lexicon_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ensure_nltk_resources()
    ensure_directories([FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR])

    analyzer = SentimentIntensityAnalyzer()
    results_df = df.copy()
    results_df["text_for_vader"] = results_df["text"].map(text_for_vader)
    results_df["text_for_textblob"] = results_df["text"].map(text_for_textblob)
    results_df["text_for_sentiwordnet"] = results_df["text"].map(text_for_sentiwordnet)

    vader_results = results_df["text_for_vader"].map(
        lambda value: vader_predict(value, analyzer)
    )
    blob_results = results_df["text_for_textblob"].map(textblob_predict)
    swn_results = results_df["text_for_sentiwordnet"].map(sentiwordnet_predict)

    results_df["vader_score"] = [item[0] for item in vader_results]
    results_df["vader_pred"] = [item[1] for item in vader_results]
    results_df["textblob_score"] = [item[0] for item in blob_results]
    results_df["textblob_pred"] = [item[1] for item in blob_results]
    results_df["sentiwordnet_score"] = [item[0] for item in swn_results]
    results_df["sentiwordnet_pred"] = [item[1] for item in swn_results]

    metrics_by_model = {
        "vader": compute_classification_metrics(
            pd.Series(results_df["label"]), pd.Series(results_df["vader_pred"])
        ),
        "textblob": compute_classification_metrics(
            pd.Series(results_df["label"]), pd.Series(results_df["textblob_pred"])
        ),
        "sentiwordnet": compute_classification_metrics(
            pd.Series(results_df["label"]), pd.Series(results_df["sentiwordnet_pred"])
        ),
    }

    summary_df = pd.DataFrame(
        [
            metrics_row("VADER", metrics_by_model["vader"]),
            metrics_row("TextBlob", metrics_by_model["textblob"]),
            metrics_row("SentiWordNet", metrics_by_model["sentiwordnet"]),
        ]
    )

    return results_df, summary_df, metrics_by_model


def save_phase1_outputs(
    results_df: pd.DataFrame, summary_df: pd.DataFrame, metrics_by_model: dict
) -> None:
    predictions_path = PREDICTIONS_DIR / "phase1_lexicon_predictions.csv"
    table_path = TABLES_DIR / "phase1_lexicon_summary.csv"
    results_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(table_path, index=False)

    for key, metrics in metrics_by_model.items():
        write_json(metrics, METRICS_DIR / f"phase1_{key}_metrics.json")

    save_confusion_matrix(
        pd.Series(results_df["label"]),
        pd.Series(results_df["vader_pred"]),
        "Phase 1 VADER Confusion Matrix",
        FIGURES_DIR / "phase1_vader_confusion_matrix.png",
    )
    save_confusion_matrix(
        pd.Series(results_df["label"]),
        pd.Series(results_df["textblob_pred"]),
        "Phase 1 TextBlob Confusion Matrix",
        FIGURES_DIR / "phase1_textblob_confusion_matrix.png",
    )
    save_confusion_matrix(
        pd.Series(results_df["label"]),
        pd.Series(results_df["sentiwordnet_pred"]),
        "Phase 1 SentiWordNet Confusion Matrix",
        FIGURES_DIR / "phase1_sentiwordnet_confusion_matrix.png",
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    x_positions = range(4)
    width = 0.25
    vader_values = [
        metrics_by_model["vader"]["accuracy"],
        metrics_by_model["vader"]["precision_weighted"],
        metrics_by_model["vader"]["recall_weighted"],
        metrics_by_model["vader"]["f1_weighted"],
    ]
    textblob_values = [
        metrics_by_model["textblob"]["accuracy"],
        metrics_by_model["textblob"]["precision_weighted"],
        metrics_by_model["textblob"]["recall_weighted"],
        metrics_by_model["textblob"]["f1_weighted"],
    ]
    sentiwordnet_values = [
        metrics_by_model["sentiwordnet"]["accuracy"],
        metrics_by_model["sentiwordnet"]["precision_weighted"],
        metrics_by_model["sentiwordnet"]["recall_weighted"],
        metrics_by_model["sentiwordnet"]["f1_weighted"],
    ]
    axis.bar(
        [position - width for position in x_positions],
        vader_values,
        width,
        label="VADER",
        color="steelblue",
    )
    axis.bar(list(x_positions), textblob_values, width, label="TextBlob", color="coral")
    axis.bar(
        [position + width for position in x_positions],
        sentiwordnet_values,
        width,
        label="SentiWordNet",
        color="lightgreen",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("VADER vs TextBlob vs SentiWordNet - Performance Comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_chart.png", dpi=150)
    plt.close(fig)


def run_phase1_pipeline() -> pd.DataFrame:
    source_path = resolve_small_raw_data_path()
    raw_loaded_df = load_amazon_reviews(source_path)
    before_duplicate_removal_df = prepare_dataset(
        raw_loaded_df, remove_exact_duplicates=False
    )
    prepared_df, profile, _ = prepare_and_save_dataset(
        source_path,
        dataset_name="amazon_appliances_reviews",
    )
    save_phase1_exploration_outputs(before_duplicate_removal_df, prepared_df)
    results_df, summary_df, metrics_by_model = run_lexicon_models(prepared_df)
    save_phase1_outputs(results_df, summary_df, metrics_by_model)
    write_json(profile, METRICS_DIR / "dataset_profile.json")
    return summary_df


if __name__ == "__main__":
    summary = run_phase1_pipeline()
    print("Phase 1 lexicon comparison complete.")
    print(summary.to_string(index=False))
