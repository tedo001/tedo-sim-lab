"""Classification metrics from a confusion matrix (numpy only, so they are easy to test)."""

from __future__ import annotations

import numpy as np

__all__ = ["confusion_matrix", "per_class", "summary"]


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """``matrix[true, predicted]`` counts."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)), 1)
    return matrix


def summary(matrix: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-averaged precision, recall and F1. A class that was never predicted
    (or never present) counts as 0 for precision (or recall), as scikit-learn does with
    ``zero_division=0``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    hits = np.diag(matrix)
    predicted, actual = matrix.sum(axis=0), matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, hits / predicted, 0.0)
        recall = np.where(actual > 0, hits / actual, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"acc": float(hits.sum() / total) if total else 0.0, "precision": float(precision.mean()),
            "recall": float(recall.mean()), "f1": float(f1.mean())}


def per_class(matrix: np.ndarray) -> list[dict[str, float]]:
    """Precision, recall, F1 and support for every class (rows of ``matrix`` are the truth)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    hits, predicted, actual = np.diag(matrix), matrix.sum(axis=0), matrix.sum(axis=1)
    rows = []
    for index in range(len(matrix)):
        precision = hits[index] / predicted[index] if predicted[index] else 0.0
        recall = hits[index] / actual[index] if actual[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"precision": float(precision), "recall": float(recall), "f1": float(f1),
                     "support": float(actual[index])})
    return rows
