import json

import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR
from src.evaluate import save_confusion_matrix
from src.lexicon_baselines import run_lexicon_models
from src.utils import ensure_directories, write_json


def load_comparison_subset() -> tuple[pd.DataFrame, dict]:
    subset_path = PREDICTIONS_DIR / "phase2_lexicon_comparison_subset.csv"
    metadata_path = METRICS_DIR / "phase2_comparison_subset_metadata.json"
    if not subset_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Phase 2 comparison subset artifacts were not found. Run `uv run python -m src.prepare_phase2` first."
        )
    subset_df = pd.read_csv(subset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return subset_df, metadata


def main() -> None:
    ensure_directories([FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR])
    subset_df, subset_metadata = load_comparison_subset()
    results_df, summary_df, metrics_by_model = run_lexicon_models(subset_df)

    results_df.to_csv(
        PREDICTIONS_DIR / "phase2_test_lexicon_predictions.csv", index=False
    )
    summary_df.to_csv(TABLES_DIR / "phase2_lexicon_comparison_summary.csv", index=False)

    for key, metrics in metrics_by_model.items():
        prediction_column = f"{key}_pred"
        prediction_frame = subset_df[
            ["overall", "summary", "reviewText", "text", "label"]
        ].copy()
        prediction_frame[prediction_column] = results_df[prediction_column]
        prediction_frame.to_csv(
            PREDICTIONS_DIR / f"phase2_{key}_comparison_predictions.csv",
            index=False,
        )
        write_json(
            {
                "model": key,
                "evaluation_scope": "shared_comparison_subset",
                "subset_metadata": subset_metadata,
                "metrics": metrics,
            },
            METRICS_DIR / f"phase2_{key}_comparison_metrics.json",
        )
        save_confusion_matrix(
            results_df["label"],
            results_df[prediction_column],
            f"Phase 2 {key.title()} Comparison Confusion Matrix",
            FIGURES_DIR / f"phase2_{key}_comparison_confusion_matrix.png",
        )

    print("Phase 2 lexicon comparison complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
