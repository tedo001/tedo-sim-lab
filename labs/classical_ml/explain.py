"""Optional SHAP values for a fitted pipeline: how much each (encoded) feature moved the
model's output, on average, over a sample of test rows. Used only when the ``shap`` package
(MIT) is installed in the experiment Python, and only for tree and linear models, where it is
exact and fast."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.common.optional import is_installed

__all__ = ["SHAP_MODELS", "shap_summary"]

#: Models with a fast, exact SHAP explainer.
SHAP_MODELS = frozenset({"decision_tree", "random_forest", "gradient_boosting", "xgboost",
                         "logistic_regression", "linear_regression"})


def shap_summary(pipeline: Any, x_test: Any, model_id: str, *, rows: int = 200,
                 seed: int = 42) -> list[list[Any]] | None:
    """``[[feature, mean |SHAP|], ...]`` sorted, largest first; ``None`` when SHAP is not
    installed or the model has no fast explainer."""
    if model_id not in SHAP_MODELS or not is_installed("shap"):
        return None
    import shap

    prepare, model = pipeline[:-1], pipeline[-1]
    sample = x_test.sample(min(rows, len(x_test)), random_state=seed)
    encoded = np.asarray(prepare.transform(sample), dtype=float)
    try:
        names = [str(name) for name in prepare.get_feature_names_out()]
    except (AttributeError, ValueError):
        names = [f"feature {i}" for i in range(encoded.shape[1])]
    if model_id in ("logistic_regression", "linear_regression"):
        explainer = shap.LinearExplainer(model, encoded)
    else:
        explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(encoded))
    if values.ndim == 3:  # (rows, features, classes) or (classes, rows, features)
        values = np.abs(values).mean(axis=2 if values.shape[:2] == encoded.shape else 0)
    strength = np.abs(values).mean(axis=0)
    rows_out = [[name, float(value)] for name, value in zip(names, strength, strict=True)]
    return sorted(rows_out, key=lambda row: row[1], reverse=True)
