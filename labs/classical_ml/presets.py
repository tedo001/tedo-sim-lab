"""Ready-made classical-ML experiments for the Classical ML lab. Each opens in the Experiment
Builder, where every setting can still be changed. All run in seconds on a CPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.common.vocab import Task
from core.experiment_engine.spec import (
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    RuntimeSpec,
    SearchSpec,
    SklearnTrainingSpec,
    TransformSpec,
)

__all__ = ["CLASSICAL_PRESETS", "ClassicalPreset"]


@dataclass(frozen=True)
class ClassicalPreset:
    id: str
    title: str
    summary: str
    dataset: str
    make: Callable[[], ExperimentSpec]


def _spec(name: str, task: Task, dataset: str, model: str, *, scale: bool = False,
          params: dict[str, Any] | None = None, cv_folds: int | None = None,
          search: SearchSpec | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        name=name, task=task,
        data=DataSpec(dataset=dataset,
                      preprocessing=(TransformSpec(name="standardize"),) if scale else ()),
        model=ModelSpec(model=model, params=params or {}),
        training=SklearnTrainingSpec(cv_folds=cv_folds, search=search),
        runtime=RuntimeSpec(device="cpu"))


CLASSICAL_PRESETS: tuple[ClassicalPreset, ...] = (
    ClassicalPreset("iris_logistic", "Iris · Logistic regression",
                    "The classic first model: three species from four measurements, standardised, with "
                    "5-fold cross-validation.", "iris",
                    lambda: _spec("iris-logistic-regression", Task.TABULAR_CLASSIFICATION, "iris",
                                  "logistic_regression", scale=True, cv_folds=5)),
    ClassicalPreset("breast_cancer_xgboost", "Breast Cancer · XGBoost",
                    "Gradient-boosted trees on 30 features of cell nuclei; ROC curves, feature "
                    "importance and, with SHAP installed, SHAP values.", "breast_cancer",
                    lambda: _spec("breast-cancer-xgboost", Task.TABULAR_CLASSIFICATION, "breast_cancer",
                                  "xgboost", params={"n_estimators": 200, "max_depth": 4,
                                                     "learning_rate": 0.1})),
    ClassicalPreset("wine_svm_search", "Wine · SVM with grid search",
                    "A kernel SVM whose C and gamma are chosen by a 5-fold grid search.", "wine",
                    lambda: _spec("wine-svm-grid-search", Task.TABULAR_CLASSIFICATION, "wine", "svm",
                                  scale=True, search=SearchSpec(method="grid", space={
                                      "C": (0.1, 1, 10, 100), "gamma": ("scale", 0.01, 0.1)}))),
    ClassicalPreset("diabetes_forest", "Diabetes · Random forest",
                    "Regression: disease progression from ten baseline measurements; predicted against "
                    "actual and residuals.", "diabetes",
                    lambda: _spec("diabetes-random-forest", Task.TABULAR_REGRESSION, "diabetes",
                                  "random_forest", params={"n_estimators": 300}, cv_folds=5)),
    ClassicalPreset("iris_kmeans", "Iris · k-means",
                    "Clustering without labels; the species are only used afterwards to see how well "
                    "the clusters match them.", "iris",
                    lambda: _spec("iris-kmeans", Task.CLUSTERING, "iris", "kmeans", scale=True,
                                  params={"n_clusters": 3})),
    ClassicalPreset("digits_pca", "Digits · PCA",
                    "64 pixel values squeezed into two components, coloured by digit.", "digits",
                    lambda: _spec("digits-pca", Task.DIMENSIONALITY_REDUCTION, "digits", "pca",
                                  scale=True, params={"n_components": 2})),
)
