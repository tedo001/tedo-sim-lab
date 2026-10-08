"""From a tabular dataset and a spec to the rows a model sees: the target column, the feature
columns, an optional row cap, encoded class labels, and a train/test split that the same seed
always reproduces (training and later re-scoring both call :func:`split_table`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.common.vocab import Task
from core.dataset_registry.adapter import TabularAdapter
from core.experiment_engine.spec import ExperimentSpec, SklearnTrainingSpec

__all__ = ["SUPERVISED", "Table", "load_table", "split_table"]

SUPERVISED = frozenset({Task.TABULAR_CLASSIFICATION, Task.TABULAR_REGRESSION})


@dataclass
class Table:
    features: Any  # pandas.DataFrame
    #: Class codes 0..k-1 (classification), numbers (regression), or None.
    target: np.ndarray | None
    target_name: str | None
    #: Class names in code order (classification only).
    classes: list[str]
    #: The dataset's own target, kept for unsupervised tasks to compare against, and its names.
    truth: np.ndarray | None = None
    truth_names: list[str] | None = None


def _encode(values: Any, names: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Codes 0..k-1 in sorted order; integer labels 0..k-1 keep the dataset's class names."""
    import pandas as pd

    categories = pd.Categorical(values)
    labels = list(categories.categories)
    codes = np.asarray(categories.codes, dtype=np.int64)
    if names and len(names) == len(labels) and labels == list(range(len(labels))):
        return codes, list(names)
    return codes, [str(label) for label in labels]


def load_table(spec: ExperimentSpec, adapter: TabularAdapter, root: Path) -> Table:
    import pandas as pd

    training = spec.training
    assert isinstance(training, SklearnTrainingSpec)
    features, target = adapter.load_frame(root)
    if training.target:
        frame = features if target is None else features.join(target)
        if training.target not in frame.columns:
            raise KeyError(f"{adapter.card.name} has no column {training.target!r}")
        target, features = frame[training.target], frame.drop(columns=[training.target])
    if training.features is not None:
        wanted = [name for name in training.features if name != training.target]
        missing = [name for name in wanted if name not in features.columns]
        if missing:
            raise KeyError(f"{adapter.card.name} has no column(s) {', '.join(map(repr, missing))}")
        features = features[wanted]
    if features.shape[1] == 0:
        raise ValueError("no feature columns are left to learn from")
    supervised = spec.task in SUPERVISED
    if supervised and target is None:
        raise ValueError(f"{adapter.card.name} has no target column; name one in training.target")
    if target is not None:
        keep = target.notna().to_numpy()
        features, target = features[keep], target[keep]
    if spec.data.subset and spec.data.subset < len(features):
        rows = np.random.default_rng(spec.seed).choice(len(features), spec.data.subset, replace=False)
        rows.sort()
        features = features.iloc[rows]
        target = None if target is None else target.iloc[rows]
    features = features.reset_index(drop=True)
    names = getattr(adapter, "class_names", lambda: ())()
    if spec.task == Task.TABULAR_CLASSIFICATION:
        codes, classes = _encode(target, names)
        return Table(features, codes, str(target.name), classes)
    if spec.task == Task.TABULAR_REGRESSION:
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(f"regression needs a numeric target; {target.name!r} holds text")
        return Table(features, target.to_numpy(dtype=float), str(target.name), [])
    truth, truth_names = None, None
    if target is not None and (not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 50):
        truth, truth_names = _encode(target, names)
    return Table(features, None, None if target is None else str(target.name), [], truth, truth_names)


def split_table(spec: ExperimentSpec, table: Table) -> tuple[Any, Any, Any, Any]:
    """``(x_train, x_test, y_train, y_test)``, stratified for classification when every class has
    at least two rows."""
    from sklearn.model_selection import train_test_split

    training = spec.training
    assert isinstance(training, SklearnTrainingSpec) and table.target is not None
    stratify = None
    if spec.task == Task.TABULAR_CLASSIFICATION and spec.data.split.stratify:
        counts = np.bincount(table.target)
        stratify = table.target if counts[counts > 0].min() >= 2 else None
    return train_test_split(table.features, table.target, test_size=training.test_fraction,
                            random_state=spec.seed, stratify=stratify)
