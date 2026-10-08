"""What a fitted classical model is measured by: metrics, curves, feature importance, and a
2-D view of clusters or projections. Everything is plain lists, ready for JSON."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["classification_report", "cluster_report", "importance", "projection_report", "regression_report"]

_MAX_POINTS = 200


def _thin(x: np.ndarray, y: np.ndarray, limit: int = _MAX_POINTS) -> list[list[float]]:
    """At most ``limit`` (x, y) pairs, always keeping the first and last."""
    index = np.unique(np.linspace(0, len(x) - 1, min(limit, len(x))).round().astype(int))
    return [[float(x[i]), float(y[i])] for i in index]


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
                          classes: list[str], scores: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy and macro P/R/F1 always; log loss from probabilities; ROC and precision-recall
    curves per class from probabilities or, for models without them, decision scores."""
    from sklearn import metrics

    labels = list(range(len(classes)))
    report: dict[str, Any] = {
        "metrics": {"acc": metrics.accuracy_score(y_true, y_pred),
                    "precision": metrics.precision_score(y_true, y_pred, labels=labels, average="macro",
                                                         zero_division=0),
                    "recall": metrics.recall_score(y_true, y_pred, labels=labels, average="macro",
                                                   zero_division=0),
                    "f1": metrics.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)},
        "matrix": metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": classes, "curves": [],
    }
    if proba is not None:
        report["metrics"]["log_loss"] = metrics.log_loss(y_true, proba, labels=labels)
    ranking = proba if proba is not None else scores
    if ranking is not None:
        if ranking.ndim == 1:  # binary decision scores: positive = class 1
            ranking = np.stack([-ranking, ranking], axis=1)
        present = [c for c in labels if (y_true == c).any() and (y_true != c).any()]
        aucs = []
        for c in present:
            fpr, tpr, _ = metrics.roc_curve(y_true == c, ranking[:, c])
            precision, recall, _ = metrics.precision_recall_curve(y_true == c, ranking[:, c])
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            report["curves"].append({"class": classes[c], "roc": _thin(fpr, tpr), "roc_auc": float(auc),
                                     "pr": _thin(recall[::-1], precision[::-1]),
                                     "average_precision": float(metrics.average_precision_score(
                                         y_true == c, ranking[:, c]))})
        if aucs:
            report["metrics"]["roc_auc"] = float(np.mean(aucs))
    report["metrics"] = {key: float(value) for key, value in report["metrics"].items()}
    return report


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn import metrics

    residuals = y_true - y_pred
    order = np.random.default_rng(0).permutation(len(y_true))[:500]
    return {"metrics": {"r2": float(metrics.r2_score(y_true, y_pred)),
                        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
                        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
                        "residual_std": float(np.std(residuals))},
            "predicted_vs_actual": [[float(y_true[i]), float(y_pred[i])] for i in order]}


def importance(pipeline: Any, x_test: Any, y_test: Any, scoring: str | None, seed: int) -> list[list[Any]]:
    """Permutation importance on the original columns: how much the test score drops when a
    column is shuffled. Works the same for every model. Sorted, largest first."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(pipeline, x_test, y_test, n_repeats=5, random_state=seed, scoring=scoring)
    rows = [[str(name), float(mean), float(std)] for name, mean, std
            in zip(x_test.columns, result.importances_mean, result.importances_std, strict=True)]
    return sorted(rows, key=lambda row: row[1], reverse=True)


def _project(features: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    if features.shape[1] <= 2:
        return np.pad(features, ((0, 0), (0, 2 - features.shape[1])))
    return PCA(n_components=2, random_state=seed).fit_transform(features)


def cluster_report(model: Any, features: np.ndarray, labels: np.ndarray, truth: np.ndarray | None,
                   seed: int) -> dict[str, Any]:
    from sklearn import metrics

    found = {"clusters": int(len(set(labels.tolist()))),
             "silhouette": float(metrics.silhouette_score(features, labels)) if len(set(labels.tolist())) > 1
             else 0.0}
    if hasattr(model, "inertia_"):
        found["inertia"] = float(model.inertia_)
    if truth is not None:
        found["adjusted_rand"] = float(metrics.adjusted_rand_score(truth, labels))
    points = _project(features, seed)
    order = np.random.default_rng(seed).permutation(len(points))[:1000]
    return {"metrics": found, "sizes": np.bincount(labels).tolist(),
            "points": [[float(points[i, 0]), float(points[i, 1]), int(labels[i])] for i in order]}


def projection_report(model: Any, projected: np.ndarray, truth: np.ndarray | None,
                      seed: int) -> dict[str, Any]:
    ratio = getattr(model, "explained_variance_ratio_", None)
    order = np.random.default_rng(seed).permutation(len(projected))[:1000]
    group = truth if truth is not None else np.zeros(len(projected), dtype=int)
    found = {"components": int(projected.shape[1])}
    if ratio is not None:
        found["explained_variance"] = float(np.sum(ratio))
    return {"metrics": found, "explained_variance_ratio": [] if ratio is None else [float(r) for r in ratio],
            "points": [[float(projected[i, 0]), float(projected[i, 1] if projected.shape[1] > 1 else 0.0),
                        int(group[i])] for i in order]}
