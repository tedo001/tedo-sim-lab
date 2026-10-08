"""scikit-learn and XGBoost estimators for the cards in ``configs/models/classical.yaml``.

Each builder takes the task and the experiment's ``model.params`` and returns an unfitted
estimator. Defaults are scikit-learn's, with a fixed ``random_state`` where the model has
one, so a run is repeatable.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BUILDERS", "build_decision_tree", "build_gradient_boosting", "build_kmeans", "build_knn",
           "build_linear_regression", "build_logistic_regression", "build_naive_bayes", "build_pca",
           "build_random_forest", "build_svm", "build_xgboost"]

CLASSIFY, REGRESS = "tabular_classification", "tabular_regression"


def _pick(task: str, classifier: Any, regressor: Any) -> Any:
    if task == CLASSIFY:
        return classifier
    if task == REGRESS and regressor is not None:
        return regressor
    raise ValueError(f"this model does not do {task.replace('_', ' ')}")


def _seeded(params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {"random_state": seed, **params}


def build_logistic_regression(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    return _pick(task, LogisticRegression, None)(**_seeded({"max_iter": 2000, **params}, seed))


def build_linear_regression(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.linear_model import LinearRegression

    if task != REGRESS:
        raise ValueError("linear regression predicts numbers: use it for tabular regression")
    return LinearRegression(**params)


def build_decision_tree(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return _pick(task, DecisionTreeClassifier, DecisionTreeRegressor)(**_seeded(params, seed))


def build_random_forest(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    return _pick(task, RandomForestClassifier, RandomForestRegressor)(**_seeded(params, seed))


def build_gradient_boosting(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    return _pick(task, GradientBoostingClassifier, GradientBoostingRegressor)(**_seeded(params, seed))


def build_xgboost(*, task: str, seed: int = 42, **params: Any) -> Any:
    from xgboost import XGBClassifier, XGBRegressor

    return _pick(task, XGBClassifier, XGBRegressor)(**_seeded({"n_estimators": 200, "tree_method": "hist",
                                                               **params}, seed))


def build_svm(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.svm import SVC, SVR

    if task == CLASSIFY:  # no probability=True (deprecated): ROC/PR use decision_function scores
        return SVC(**_seeded(params, seed))
    return _pick(task, None, SVR)(**params)


def build_knn(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    return _pick(task, KNeighborsClassifier, KNeighborsRegressor)(**params)


def build_naive_bayes(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.naive_bayes import GaussianNB

    return _pick(task, GaussianNB, None)(**params)


def build_pca(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.decomposition import PCA

    if task != "dimensionality_reduction":
        raise ValueError("PCA reduces dimensions: use it for dimensionality reduction")
    return PCA(**_seeded({"n_components": 2, **params}, seed))


def build_kmeans(*, task: str, seed: int = 42, **params: Any) -> Any:
    from sklearn.cluster import KMeans

    if task != "clustering":
        raise ValueError("k-means groups rows: use it for clustering")
    return KMeans(**_seeded({"n_clusters": 3, "n_init": "auto", **params}, seed))


BUILDERS = {name.removeprefix("build_"): value for name, value in dict(globals()).items()
            if name.startswith("build_")}
