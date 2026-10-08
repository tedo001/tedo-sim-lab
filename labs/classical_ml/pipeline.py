"""From a spec to a scikit-learn ``Pipeline``: imputation and one-hot encoding always (so any
model accepts any table), scaling and feature selection when asked, the model last, and a
grid or random search around it when the spec has one."""

from __future__ import annotations

from typing import Any

from core.experiment_engine.spec import ExperimentSpec, SklearnTrainingSpec

__all__ = ["PREPROCESSING", "build_pipeline", "pipeline_problems", "search_space", "wrap_search"]

#: name → what it does (the Experiment Builder shows these).
PREPROCESSING = {
    "standardize": "Scale numeric columns to mean 0 and standard deviation 1",
    "minmax": "Scale numeric columns to 0 … 1",
    "select_k_best": "Keep the k features most related to the target (params: k)",
}


def pipeline_problems(spec: ExperimentSpec) -> list[str]:
    problems = [f"unknown preprocessing step {step.name!r} (known: {', '.join(PREPROCESSING)})"
                for step in spec.data.preprocessing if step.name not in PREPROCESSING]
    if spec.data.augmentation:
        problems.append("augmentation is for images; tabular experiments have none")
    names = [step.name for step in spec.data.preprocessing]
    if "standardize" in names and "minmax" in names:
        problems.append("choose one scaling: standardize or minmax")
    return problems


def build_pipeline(spec: ExperimentSpec, model: Any, columns: Any, *, supervised: bool) -> Any:
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

    steps = {step.name: step for step in spec.data.preprocessing}
    numeric: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if "standardize" in steps:
        numeric.append(("scale", StandardScaler()))
    elif "minmax" in steps:
        numeric.append(("scale", MinMaxScaler()))
    text = [("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    prepare = ColumnTransformer([
        ("numeric", Pipeline(numeric), make_column_selector(dtype_include="number")),
        ("text", Pipeline(text), make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
    stages: list[tuple[str, Any]] = [("prepare", prepare)]
    if "select_k_best" in steps and supervised:
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression

        score = f_classif if spec.task.value == "tabular_classification" else f_regression
        k = steps["select_k_best"].params.get("k", 10)
        stages.append(("select", SelectKBest(score, k=min(int(k), len(columns)) if k != "all" else "all")))
    stages.append(("model", model))
    return Pipeline(stages)


def search_space(training: SklearnTrainingSpec) -> dict[str, list[Any]]:
    """The search space with every key aimed at the model step (``C`` → ``model__C``)."""
    assert training.search is not None
    return {(key if "__" in key else f"model__{key}"): list(values)
            for key, values in training.search.space.items()}


def wrap_search(pipeline: Any, training: SklearnTrainingSpec, seed: int, scoring: str | None) -> Any:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    search = training.search
    assert search is not None
    folds = training.cv_folds or 5
    if search.method == "random":
        return RandomizedSearchCV(pipeline, search_space(training), n_iter=search.n_iter or 10, cv=folds,
                                  scoring=search.scoring or scoring, random_state=seed, n_jobs=1)
    return GridSearchCV(pipeline, search_space(training), cv=folds, scoring=search.scoring or scoring,
                        n_jobs=1)
