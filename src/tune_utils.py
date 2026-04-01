from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_RANDOM_STATE, METRICS_DIR, TABLES_DIR
from src.prepare_phase2 import prepare_phase2_artifacts
from src.utils import ensure_directories, write_json


def load_phase2_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = prepare_phase2_artifacts()
    development_df = cast(pd.DataFrame, outputs["development_df"])
    train_df, test_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            development_df,
            test_size=0.3,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=development_df["overall"],
        ),
    )
    return train_df, test_df


def run_grid_search(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    ensure_directories([TABLES_DIR, METRICS_DIR])
    train_df, _ = load_phase2_splits()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE),
        n_jobs=None,
        refit=True,
        return_train_score=False,
    )
    search.fit(cast(pd.Series, train_df["text"]), cast(pd.Series, train_df["label"]))

    results_df = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
    results_df.to_csv(TABLES_DIR / f"tuning_{model_name}.csv", index=False)
    write_json(
        {
            "model": model_name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
        METRICS_DIR / f"tuning_{model_name}.json",
    )
    return pd.DataFrame(
        results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    )
