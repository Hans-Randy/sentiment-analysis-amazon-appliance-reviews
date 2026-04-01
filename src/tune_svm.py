from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DEFAULT_RANDOM_STATE
from src.features import build_tfidf_vectorizer
from src.tune_utils import run_grid_search


def main() -> None:
    pipeline = Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )
    results = run_grid_search(
        model_name="svm",
        pipeline=pipeline,
        param_grid={"tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
    )
    print("Linear SVC tuning complete.")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
