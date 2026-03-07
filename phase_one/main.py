# %% [markdown]
# # IMPORTS

# %%
import json
import gzip
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# %%
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# %%
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# %% [markdown]
# # LOAD DATA

# %%
DATA_PATH = "data/Appliances_5.json.gz"

def load_amazon_gz(path):
    """Load a gzipped JSON-lines file into a DataFrame."""
    records = []
    with gzip.open(path, "rb") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


df = load_amazon_gz(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} reviews, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}\n")

# %% [markdown]
# # DATA EXPLORATION

# %%
print("\n--- Basic Info ---")
print(df.dtypes)
print(f"\nTotal reviews       : {len(df)}")
print(f"Unique products     : {df['asin'].nunique()}")
print(f"Unique reviewers    : {df['reviewerID'].nunique()}")
print(f"Date range          : {df['reviewTime'].min()} – {df['reviewTime'].max()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# %%
print("\n--- Rating Distribution ---")
print(df["overall"].value_counts().sort_index())

# %%
fig, ax = plt.subplots(figsize=(6, 4))
df["overall"].value_counts().sort_index().plot(kind="bar", color="steelblue", ax=ax)
ax.set_title("Distribution of Ratings")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png", dpi=150)
plt.show()

# %%
reviews_per_product = df.groupby("asin").size()
print("\n--- Reviews Per Product ---")
print(reviews_per_product.describe())

# %%
fig, ax = plt.subplots(figsize=(6, 4))
reviews_per_product.hist(bins=30, color="coral", edgecolor="black", ax=ax)
ax.set_title("Distribution of Reviews Per Product")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Number of Products")
plt.tight_layout()
plt.savefig("outputs/reviews_per_product.png", dpi=150)
plt.show()

# %%
reviews_per_user = df.groupby("reviewerID").size()
print("\n--- Reviews Per User ---")
print(reviews_per_user.describe())

# %%
fig, ax = plt.subplots(figsize=(6, 4))
reviews_per_user.hist(bins=30, color="mediumpurple", edgecolor="black", ax=ax)
ax.set_title("Distribution of Reviews Per User")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Number of Users")
plt.tight_layout()
plt.savefig("outputs/reviews_per_user.png", dpi=150)
plt.show()

# %%
df["reviewText"] = df["reviewText"].fillna("")
df["review_length"] = df["reviewText"].apply(len)
df["review_word_count"] = df["reviewText"].apply(lambda x: len(x.split()))

print("\n--- Review Length (characters) ---")
print(df["review_length"].describe())

print("\n--- Review Word Count ---")
print(df["review_word_count"].describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["review_length"].hist(bins=50, color="teal", edgecolor="black", ax=axes[0])
axes[0].set_title("Review Length (Characters)")
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency")

df["review_word_count"].hist(bins=50, color="salmon", edgecolor="black", ax=axes[1])
axes[1].set_title("Review Length (Words)")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/review_lengths.png", dpi=150)
plt.show()

# %%
q1 = df["review_word_count"].quantile(0.25)
q3 = df["review_word_count"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[
    (df["review_word_count"] < lower_bound)
    | (df["review_word_count"] > upper_bound)
]
print(f"\n--- Outliers (IQR method on word count) ---")
print(f"Lower bound: {lower_bound:.0f}, Upper bound: {upper_bound:.0f}")
print(f"Number of outlier reviews: {len(outliers)}")

fig, ax = plt.subplots(figsize=(6, 4))
df.boxplot(column="review_word_count", ax=ax)
ax.set_title("Boxplot of Review Word Count")
plt.tight_layout()
plt.savefig("outputs/review_wordcount_boxplot.png", dpi=150)
plt.show()

# %%
duplicate_count = df.duplicated(subset=["reviewerID", "asin", "reviewText"]).sum()
print(f"\n--- Duplicates ---")
print(f"Duplicate reviews (same user, product, text): {duplicate_count}")

# Remove exact duplicates if any
df = df.drop_duplicates(subset=["reviewerID", "asin", "reviewText"]).reset_index(
    drop=True
)
print(f"Dataset after duplicate removal: {len(df)} reviews")

# Average rating
print(f"\n--- Averages ---")
print(f"Mean rating: {df['overall'].mean():.2f}")
print(f"Median rating: {df['overall'].median():.1f}")

# Verified vs unverified
if "verified" in df.columns:
    print(f"\nVerified purchases: {df['verified'].sum()}")
    print(f"Unverified purchases: {(~df['verified']).sum()}")

# %% [markdown]
# # TEXT PRE-PROCESSING & LABELING
# 
# Label based on rating
# We map the star rating to sentiment classes to create ground truth labels for evaluating lexicon-based classifiers.

# %%
def label_sentiment(rating):
    """Map numeric rating to sentiment label."""
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


df["sentiment"] = df["overall"].apply(label_sentiment)
print("\n--- Sentiment Label Distribution ---")
print(df["sentiment"].value_counts())

fig, ax = plt.subplots(figsize=(5, 4))
df["sentiment"].value_counts().plot(
    kind="bar",
    color=["green", "gold", "red"],
    ax=ax,
)
ax.set_title("Sentiment Label Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# We choose 'reviewText' as the primary column for sentiment analysis because it contains the full customer opinion.  
# We also keep 'summary' as it provides a condensed version that can reinforce sentiment signals.
# 'overall' is kept as ground truth.  'asin' and 'reviewerID' are kept for identification.

# %%
COLUMNS_USED = [
    "reviewerID",
    "asin",
    "overall",
    "reviewText",
    "summary",
    "sentiment",
    "review_word_count",
]
df = df[[c for c in COLUMNS_USED if c in df.columns]].copy()
print(f"\nColumns selected for analysis: {df.columns.tolist()}")

# %%
empty_reviews = (df["reviewText"].str.strip() == "").sum()
print(f"\nEmpty review texts: {empty_reviews}")
# Remove empty reviews
df = df[df["reviewText"].str.strip() != ""].reset_index(drop=True)
print(f"Dataset after removing empty reviews: {len(df)} reviews")

# %% [markdown]
# # TEXT PRE-PROCESSING FOR EACH LEXICON
# 
# We studied three lexicon packages:
# 
# 1. VADER (Valence Aware Dictionary and Sentiment Reasoner)
#    - Specifically tuned for social media / short informal text
#    - Handles capitalization, punctuation emphasis, slang, emojis
#    - Returns compound score (-1 to +1)
#    - Works well WITHOUT removing stop words or punctuation
# 
# 2. TextBlob
#    - Built on Pattern library and NLTK
#    - Returns polarity (-1 to +1) and subjectivity (0 to 1)
#    - General-purpose, easy to use
#    - Works well on standard English text
# 
# 3. SentiWordNet
#    - Assigns positivity/negativity/objectivity to WordNet synsets
#    - Requires POS tagging and word-sense disambiguation
#    - More complex to implement; best for fine-grained analysis
# 
# CHOSEN: VADER and TextBlob
# - VADER excels on informal review text (handles emphasis, negation well)
# - TextBlob provides a good general-purpose baseline for comparison
# - Both are straightforward to apply and compare
# - SentiWordNet was not chosen because it requires complex word-sense
#   disambiguation which adds overhead without significant benefit on
#   short product reviews.

# %%
df["text_for_vader"] = df["reviewText"].str.strip()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_for_textblob(text):
    """Clean text for TextBlob analysis."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters & spaces
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)

df["text_for_textblob"] = df["reviewText"].apply(preprocess_for_textblob)

print("\n--- Sample (first review) ---")
print(f"Original      : {df['reviewText'].iloc[0][:120]}...")
print(f"For VADER     : {df['text_for_vader'].iloc[0][:120]}...")
print(f"For TextBlob  : {df['text_for_textblob'].iloc[0][:120]}...")

# %% [markdown]
# # SAMPLE 1000 REVIEWS

# %%
SAMPLE_SIZE = min(1000, len(df))
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"Randomly selected {SAMPLE_SIZE} reviews (random_state=42)")
print(f"Sentiment distribution in sample:\n{df_sample['sentiment'].value_counts()}")

# %% [markdown]
# # MODELING – LEXICON APPROACH

# %%
print("\n--- VADER Model ---")
vader_analyzer = SentimentIntensityAnalyzer()


def vader_predict(text):
    """Classify sentiment using VADER compound score."""
    scores = vader_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df_sample["vader_pred"] = df_sample["text_for_vader"].apply(vader_predict)
df_sample["vader_compound"] = df_sample["text_for_vader"].apply(
    lambda x: vader_analyzer.polarity_scores(x)["compound"]
)

print("VADER predictions (sample):")
print(df_sample["vader_pred"].value_counts())

# %%
print("\n--- TextBlob Model ---")


def textblob_predict(text):
    """Classify sentiment using TextBlob polarity."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


df_sample["textblob_pred"] = df_sample["text_for_textblob"].apply(textblob_predict)
df_sample["textblob_polarity"] = df_sample["text_for_textblob"].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

print("TextBlob predictions (sample):")
print(df_sample["textblob_pred"].value_counts())

# %% [markdown]
# # VALIDATION & COMPARISON

# %%
labels = ["Positive", "Neutral", "Negative"]
y_true = df_sample["sentiment"]

# %% [markdown]
# ## VADER METRICS

# %%
y_vader = df_sample["vader_pred"]
print("\n===== VADER Classification Report =====")
print(
    classification_report(
        y_true, y_vader, labels=labels, target_names=labels, zero_division=0
    )
)

vader_acc = accuracy_score(y_true, y_vader)
vader_prec = precision_score(y_true, y_vader, average="weighted", zero_division=0)
vader_rec = recall_score(y_true, y_vader, average="weighted", zero_division=0)
vader_f1 = f1_score(y_true, y_vader, average="weighted", zero_division=0)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm_vader = confusion_matrix(y_true, y_vader, labels=labels)
ConfusionMatrixDisplay(cm_vader, display_labels=labels).plot(ax=ax, cmap="Blues")
ax.set_title("VADER – Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/cm_vader.png", dpi=150)
plt.show()

# %% [markdown]
# ## TextBlob Metrics

# %%
y_textblob = df_sample["textblob_pred"]
print("\n===== TextBlob Classification Report =====")
print(
    classification_report(
        y_true, y_textblob, labels=labels, target_names=labels, zero_division=0
    )
)

tb_acc = accuracy_score(y_true, y_textblob)
tb_prec = precision_score(y_true, y_textblob, average="weighted", zero_division=0)
tb_rec = recall_score(y_true, y_textblob, average="weighted", zero_division=0)
tb_f1 = f1_score(y_true, y_textblob, average="weighted", zero_division=0)

fig, ax = plt.subplots(figsize=(6, 5))
cm_tb = confusion_matrix(y_true, y_textblob, labels=labels)
ConfusionMatrixDisplay(cm_tb, display_labels=labels).plot(ax=ax, cmap="Oranges")
ax.set_title("TextBlob – Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/cm_textblob.png", dpi=150)
plt.show()

# %% [markdown]
# ## Comparison Table

# %%
print("\n===== Comparison Table =====")
comparison = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"],
        "VADER": [
            f"{vader_acc:.4f}",
            f"{vader_prec:.4f}",
            f"{vader_rec:.4f}",
            f"{vader_f1:.4f}",
        ],
        "TextBlob": [
            f"{tb_acc:.4f}",
            f"{tb_prec:.4f}",
            f"{tb_rec:.4f}",
            f"{tb_f1:.4f}",
        ],
    }
)
print(comparison.to_string(index=False))

# Side-by-side bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
width = 0.3
vals_vader = [vader_acc, vader_prec, vader_rec, vader_f1]
vals_tb = [tb_acc, tb_prec, tb_rec, tb_f1]

ax.bar(x - width / 2, vals_vader, width, label="VADER", color="steelblue")
ax.bar(x + width / 2, vals_tb, width, label="TextBlob", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("VADER vs TextBlob – Performance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/comparison_chart.png", dpi=150)
plt.show()

# %% [markdown]
# ### Score distribution plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_sample["vader_compound"], bins=40, color="steelblue", edgecolor="black")
axes[0].axvline(0.05, color="green", linestyle="--", label="Pos threshold (0.05)")
axes[0].axvline(-0.05, color="red", linestyle="--", label="Neg threshold (-0.05)")
axes[0].set_title("VADER Compound Score Distribution")
axes[0].set_xlabel("Compound Score")
axes[0].legend()

axes[1].hist(
    df_sample["textblob_polarity"], bins=40, color="coral", edgecolor="black"
)
axes[1].axvline(0.1, color="green", linestyle="--", label="Pos threshold (0.1)")
axes[1].axvline(-0.1, color="red", linestyle="--", label="Neg threshold (-0.1)")
axes[1].set_title("TextBlob Polarity Distribution")
axes[1].set_xlabel("Polarity")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/score_distributions.png", dpi=150)
plt.show()

# %%
df_sample.to_csv("outputs/phase1_results.csv", index=False)
print("\nResults saved to outputs/phase1_results.csv")
print("\nPhase #1 complete!")


