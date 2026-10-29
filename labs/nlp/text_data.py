"""The texts of a text-classification experiment: the dataset's own train/test split when it has
one, else a stratified hold-out of ``test_fraction``; ``subset`` caps the training texts (and the
test texts, so a quick run stays quick). Shared by the TF-IDF and the transformer runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.catalog import load_cards
from core.common.paths import AppPaths
from core.dataset_registry.adapter import TextAdapter
from core.experiment_engine.spec import ExperimentSpec

__all__ = ["TextSplits", "load_splits", "split_samples", "text_adapter"]


@dataclass
class TextSplits:
    train: list[str]
    y_train: np.ndarray
    test: list[str]
    y_test: np.ndarray
    classes: list[str]
    own_split: bool


def text_adapter(spec: ExperimentSpec) -> tuple[AppPaths, TextAdapter, Any]:
    paths = AppPaths.resolve()
    datasets, models = load_cards(paths)
    if spec.data.dataset not in datasets:
        raise LookupError(f"no dataset card {spec.data.dataset!r}")
    adapter = datasets.adapter(spec.data.dataset)
    if not isinstance(adapter, TextAdapter):
        raise TypeError(f"{adapter.card.name} is not a text dataset")
    if spec.model.model not in models:
        raise LookupError(f"no model card {spec.model.model!r}")
    card = models.get(spec.model.model)
    if spec.task not in card.tasks:
        raise ValueError(f"{card.name} does not do {spec.task.value.replace('_', ' ')}")
    return paths, adapter, card


def load_splits(spec: ExperimentSpec, adapter: TextAdapter, root: Path, test_fraction: float) -> TextSplits:
    classes = list(adapter.classes)
    if adapter.has_test_split:
        train, y_train = adapter.load_texts(root, "train")
        test, y_test = adapter.load_texts(root, "test")
        own = True
    else:
        from sklearn.model_selection import train_test_split

        texts, labels = adapter.load_texts(root, "all")
        train, test, y_train, y_test = train_test_split(texts, labels, test_size=test_fraction,
                                                        random_state=spec.seed, stratify=labels)
        own = False
    rng = np.random.default_rng(spec.seed)
    y_train, y_test = np.asarray(y_train, dtype=np.int64), np.asarray(y_test, dtype=np.int64)
    if spec.data.subset is not None:
        keep = np.sort(rng.permutation(len(train))[: spec.data.subset])
        train, y_train = [train[i] for i in keep], y_train[keep]
        keep = np.sort(rng.permutation(len(test))[: spec.data.subset])
        test, y_test = [test[i] for i in keep], y_test[keep]
    return TextSplits(list(train), y_train, list(test), y_test, classes, own)


def split_samples(spec: ExperimentSpec, data: TextSplits) -> dict[str, Any]:
    """The texts each split gets, for the data checks (case and spacing ignored, so a copy that
    differs only there still counts as the same text)."""
    from core.dataset_registry.validation import SampleSet, digest

    def key(text: str) -> str:
        return digest(" ".join(text.lower().split()))

    return {"train": SampleSet([key(t) for t in data.train], data.y_train.tolist(), classes=data.classes),
            "test": SampleSet([key(t) for t in data.test], data.y_test.tolist(), classes=data.classes)}
