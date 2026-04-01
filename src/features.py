from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DEFAULT_RANDOM_STATE


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=5000,
        sublinear_tf=True,
    )


def random_state() -> int:
    return DEFAULT_RANDOM_STATE
