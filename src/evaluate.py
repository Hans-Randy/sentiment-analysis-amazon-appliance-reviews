from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import LABEL_ORDER


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
    }


def metrics_row(model_name: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
    }


def save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, title: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, axis = plt.subplots(figsize=(7, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(LABEL_ORDER)))
    axis.set_yticks(range(len(LABEL_ORDER)))
    axis.set_xticklabels(LABEL_ORDER, rotation=15)
    axis.set_yticklabels(LABEL_ORDER)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)

    threshold = matrix.max() * 0.5 if matrix.size else 0

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value > threshold else "black"
            axis.text(
                col_index,
                row_index,
                value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
