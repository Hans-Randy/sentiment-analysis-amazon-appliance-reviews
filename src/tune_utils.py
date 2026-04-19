from typing import cast

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import DEFAULT_RANDOM_STATE, METRICS_DIR, TABLES_DIR
from src.model_registry import MODEL_SPECS
from src.prepare_phase2 import prepare_phase2_artifacts
from src.utils import ensure_directories, write_json
import os   # Changes - For handling file paths and directories
import shutil # Changes - For removing temporary cache directories after use


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
    scoring: str = "f1_weighted",
) -> pd.DataFrame:
    ensure_directories([TABLES_DIR, METRICS_DIR])
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model name: {model_name}")

    model_spec = MODEL_SPECS[model_name]
    train_df, _ = load_phase2_splits()

# --- 🟢 HIGHLIGHT: CHANGES START HERE 🟢 ---
    
    # 1. Grab the pipeline object first
    pipeline = model_spec.builder()
    
    # 2. Attach a cache directory to the pipeline
    # This prevents the TF-IDF math from repeating for every SVM parameter
    cache_dir = "./pipeline_cache"
    pipeline.memory = cache_dir
    search = GridSearchCV(
        # estimator=model_spec.builder(),
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE),
        # n_jobs=None,
        n_jobs=-1, # Use all available CPU cores for parallel processing
        refit=True,
        return_train_score=False,
    )
    # --- 🔴 HIGHLIGHT: CHANGES END HERE 🔴 ---

    search.fit(cast(pd.Series, train_df["text"]), cast(pd.Series, train_df["label"]))

# --- 🟢 HIGHLIGHT: OPTIONAL CACHE CLEANUP 🟢 ---
    # Delete the temporary folder after training so it doesn't clutter your project
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    # -----------------------------------------------

    results_df = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
    results_df.to_csv(TABLES_DIR / f"tuning_{model_name}.csv", index=False)
    write_json(
        {
            "model": model_spec.display_name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
        METRICS_DIR / f"tuning_{model_name}.json",
    )
    return pd.DataFrame(
        results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    )
