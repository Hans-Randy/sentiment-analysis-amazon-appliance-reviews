from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_RANDOM_STATE
from src.features import build_tfidf_vectorizer
from src.tune_utils import run_grid_search


def main() -> None:
    pipeline = Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )
    results = run_grid_search(
        model_name="logistic_regression",
        pipeline=pipeline,
        param_grid={"tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
    )
    print("Logistic regression tuning complete.")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
