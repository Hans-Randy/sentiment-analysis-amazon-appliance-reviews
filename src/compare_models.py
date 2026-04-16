import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, TABLES_DIR
from src.utils import ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate saved Phase 2 comparison metrics."
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Optional directory containing Phase 2 comparison metric JSON files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected lexicon test metric files are missing.",
    )
    return parser.parse_args()


def load_metric_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_comparison_metric_paths(metrics_dir: Path) -> list[Path]:
    return sorted(metrics_dir.glob("phase2_*_comparison_metrics.json"))


_PATH_KEYS = {"source_file", "subset_path"}


def _subset_fingerprint(metadata: dict) -> str:
    """Serialize metadata fields that identify the subset, excluding machine-specific paths."""
    subset = metadata.get("subset", {})
    fingerprint = {
        "random_state": metadata.get("random_state"),
        "development_sample_size": metadata.get("development_sample_size"),
        "train_rows": metadata.get("train_rows"),
        "test_rows": metadata.get("test_rows"),
        "train_test_split": metadata.get("train_test_split"),
        "comparison_scope": metadata.get("comparison_scope"),
        "comparison_subset_size_requested": metadata.get("comparison_subset_size_requested"),
        "subset_name": subset.get("subset_name"),
        "subset_rows": subset.get("rows"),
        "text_hash_sum": subset.get("text_hash_sum"),
        "label_distribution": subset.get("label_distribution"),
    }
    return json.dumps(fingerprint, sort_keys=True)


def validate_subset_metadata(metric_payloads: list[dict]) -> dict:
    if not metric_payloads:
        raise ValueError("No comparison metric files were found.")
    reference_metadata = metric_payloads[0]["subset_metadata"]
    reference_fingerprint = _subset_fingerprint(reference_metadata)
    for payload in metric_payloads[1:]:
        if _subset_fingerprint(payload["subset_metadata"]) != reference_fingerprint:
            raise ValueError(
                "Comparison metric files do not reference the same evaluation subset."
            )
    return reference_metadata


def metrics_row_from_payload(payload: dict) -> dict:
    metrics = payload["metrics"]
    model_name = payload["model"]
    display_name = {
        "vader": "VADER",
        "textblob": "TextBlob",
        "sentiwordnet": "SentiWordNet",
    }.get(model_name, model_name)
    return {
        "model": display_name,
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
    }


def save_comparison_chart(comparison_df: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    chart_df = comparison_df.set_index("model")[["accuracy", "f1_weighted"]]
    chart_df.plot(kind="bar", ax=axis, color=["steelblue", "darkorange"])
    axis.set_title("Phase 2 Model Comparison on Shared Evaluation Subset")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.legend(["Accuracy", "Weighted F1"])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase2_model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories([FIGURES_DIR, TABLES_DIR])
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else METRICS_DIR
    metric_paths = collect_comparison_metric_paths(metrics_dir)
    if args.strict:
        required = {
            metrics_dir / "phase2_vader_comparison_metrics.json",
            metrics_dir / "phase2_textblob_comparison_metrics.json",
            metrics_dir / "phase2_sentiwordnet_comparison_metrics.json",
        }
        missing = sorted(str(path.name) for path in required if not path.exists())
        if missing:
            raise FileNotFoundError(
                f"Missing required lexicon test metric files: {', '.join(missing)}"
            )
    payloads = [load_metric_file(path) for path in metric_paths]
    subset_metadata = validate_subset_metadata(payloads)
    comparison_df = pd.DataFrame(
        [metrics_row_from_payload(payload) for payload in payloads]
    ).sort_values(by=["f1_weighted", "accuracy"], ascending=False)
    comparison_df.to_csv(TABLES_DIR / "phase2_model_comparison.csv", index=False)
    save_comparison_chart(comparison_df)
    (METRICS_DIR / "phase2_model_comparison_metadata.json").write_text(
        json.dumps(subset_metadata, indent=2), encoding="utf-8"
    )
    print("Phase 2 all-model comparison aggregation complete.")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
