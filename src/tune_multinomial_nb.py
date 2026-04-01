from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.features import build_tfidf_vectorizer
from src.tune_utils import run_grid_search


def main() -> None:
    pipeline = Pipeline(
        steps=[("tfidf", build_tfidf_vectorizer()), ("classifier", MultinomialNB())]
    )
    results = run_grid_search(
        model_name="multinomial_nb",
        pipeline=pipeline,
        param_grid={"tfidf__min_df": [1, 2], "classifier__alpha": [0.1, 0.5, 1.0]},
    )
    print("Multinomial Naive Bayes tuning complete.")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
