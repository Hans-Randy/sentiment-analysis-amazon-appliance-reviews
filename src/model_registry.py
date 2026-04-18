from dataclasses import dataclass
from typing import Any, Callable

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from src.config import DEFAULT_RANDOM_STATE
from src.features import build_tfidf_vectorizer


@dataclass(frozen=True)
class ModelSpec:
    cli_name: str
    display_name: str
    family: str
    is_default: bool
    builder: Callable[[], Pipeline]
    param_grid: dict[str, list[Any]]


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer().set_params(min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=2.0,
                    class_weight="balanced",
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_linear_svc_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer().set_params(min_df=2)),
            (
                "classifier",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_complement_nb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer().set_params(min_df=1, max_features=10000)),
            ("classifier", ComplementNB(alpha=0.5, norm=True)),
        ]
    )


def build_mlp_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                build_tfidf_vectorizer().set_params(min_df=2, max_features=15000),
            ),
            ("svd", TruncatedSVD(n_components=300, random_state=DEFAULT_RANDOM_STATE)),
            ("normalize", Normalizer(copy=False)),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    learning_rate="adaptive",
                    max_iter=25,
                    early_stopping=False,
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                build_tfidf_vectorizer().set_params(min_df=2, max_features=20000),
            ),
            ("svd", TruncatedSVD(n_components=300, random_state=DEFAULT_RANDOM_STATE)),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        cli_name="logistic_regression",
        display_name="LogisticRegression",
        family="linear",
        is_default=True,
        builder=build_logistic_regression_pipeline,
        param_grid={"tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
    ),
    "svm": ModelSpec(
        cli_name="svm",
        display_name="LinearSVC",
        family="linear",
        is_default=True,
        builder=build_linear_svc_pipeline,
        param_grid={"tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
    ),
    "complement_nb": ModelSpec(
        cli_name="complement_nb",
        display_name="ComplementNB",
        family="bayes",
        is_default=True,
        builder=build_complement_nb_pipeline,
        param_grid={
            "tfidf__min_df": [1, 2],
            "tfidf__max_features": [5000, 10000],
            "classifier__alpha": [0.01, 0.05, 0.1, 0.5],
            "classifier__norm": [True, False],
        },
    ),
    "mlp": ModelSpec(
        cli_name="mlp",
        display_name="MLP",
        family="experimental",
        is_default=False,
        builder=build_mlp_pipeline,
        param_grid={
            "svd__n_components": [200, 300],
            "classifier__hidden_layer_sizes": [(128,), (128, 64)],
            "classifier__alpha": [0.0001, 0.001],
        },
    ),
    "gradient_boosting": ModelSpec(
        cli_name="gradient_boosting",
        display_name="GradientBoosting",
        family="experimental",
        is_default=False,
        builder=build_gradient_boosting_pipeline,
        param_grid={
            "svd__n_components": [200, 300],
            "classifier__n_estimators": [200, 300],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__max_depth": [3, 4],
            "classifier__subsample": [0.8, 1.0],
        },
    ),
}


def default_model_names() -> list[str]:
    return [name for name, spec in MODEL_SPECS.items() if spec.is_default]


def experimental_model_names() -> list[str]:
    return [name for name, spec in MODEL_SPECS.items() if not spec.is_default]


def resolve_model_names(selected_names: list[str] | None) -> list[str]:
    if not selected_names:
        return default_model_names()
    invalid_names = [name for name in selected_names if name not in MODEL_SPECS]
    if invalid_names:
        raise ValueError(f"Unknown model names: {', '.join(invalid_names)}")
    return selected_names


def build_selected_pipelines(
    selected_names: list[str] | None = None,
) -> dict[str, Pipeline]:
    resolved_names = resolve_model_names(selected_names)
    return {
        MODEL_SPECS[name].display_name: MODEL_SPECS[name].builder()
        for name in resolved_names
    }
