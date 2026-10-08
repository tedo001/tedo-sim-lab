"""Turning an experiment's ``data`` section into training, validation and test sets.

The validation set is carved out of the training split (stratified by class unless
switched off), with the evaluation transforms, never the augmentations. ``subset``
caps every split for quick runs; the cap and the seed are part of the spec, so a
quick run is as reproducible as a full one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.experiment_engine.spec import ExperimentSpec, TransformSpec

from ..datasets import VisionAdapter, targets_of

__all__ = ["AUGMENTATIONS", "PREPROCESSING", "DataBundle", "build_data", "input_size_for",
           "split_indices", "transform_problems"]

#: name → what it does, for the builder and for validation.
PREPROCESSING = {"normalize": "Subtract the dataset mean and divide by its standard deviation"}
AUGMENTATIONS = {
    "random_crop": "Pad by 4 pixels and crop back at a random offset",
    "horizontal_flip": "Mirror left–right half of the time (not for digits)",
    "rotation": "Rotate by up to ±10°",
}


@dataclass
class DataBundle:
    train: Any
    val: Any | None
    test: Any
    in_channels: int
    input_size: tuple[int, int]
    classes: tuple[str, ...]
    #: The normalisation applied, if any: (mean, std) per channel.
    normalization: tuple[tuple[float, ...], tuple[float, ...]] | None


def transform_problems(spec: ExperimentSpec) -> list[str]:
    problems = [f"unknown preprocessing step {step.name!r} (known: {', '.join(PREPROCESSING)})"
                for step in spec.data.preprocessing if step.name not in PREPROCESSING]
    problems += [f"unknown augmentation {step.name!r} (known: {', '.join(AUGMENTATIONS)})"
                 for step in spec.data.augmentation if step.name not in AUGMENTATIONS]
    return problems


def input_size_for(spec: ExperimentSpec, adapter: VisionAdapter, builder: Any) -> tuple[int, int]:
    """The spec's size, else the size the model is designed for, else the dataset's own."""
    if spec.data.input_size is not None:
        return tuple(spec.data.input_size)  # type: ignore[return-value]
    return tuple(getattr(builder, "input_size", None) or adapter.native_size)  # type: ignore[return-value]


Stats = tuple[tuple[float, ...], tuple[float, ...]]


def _normalization(step: TransformSpec, adapter: VisionAdapter) -> Stats:
    mean = tuple(step.params.get("mean", adapter.mean))
    std = tuple(step.params.get("std", adapter.std))
    return mean, std


def _transforms(spec: ExperimentSpec, adapter: VisionAdapter, size: tuple[int, int], *, train: bool) -> Any:
    from torchvision import transforms as T

    steps: list[Any] = []
    if size != tuple(adapter.native_size):
        steps.append(T.Resize(size, antialias=True))
    if train:
        for step in spec.data.augmentation:
            if step.name == "random_crop":
                steps.append(T.RandomCrop(size, padding=int(step.params.get("padding", 4))))
            elif step.name == "horizontal_flip":
                steps.append(T.RandomHorizontalFlip(float(step.params.get("p", 0.5))))
            elif step.name == "rotation":
                steps.append(T.RandomRotation(float(step.params.get("degrees", 10))))
    steps.append(T.ToTensor())
    for step in spec.data.preprocessing:
        if step.name == "normalize":
            mean, std = _normalization(step, adapter)
            steps.append(T.Normalize(mean, std))
    return T.Compose(steps)


def split_indices(targets: np.ndarray, val_fraction: float, *, stratify: bool,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation indices; per class when ``stratify``, so both keep the class mix."""
    rng = np.random.default_rng(seed)
    if val_fraction <= 0:
        return rng.permutation(len(targets)), np.array([], dtype=np.int64)
    if stratify:
        groups = [np.flatnonzero(targets == c) for c in np.unique(targets)]
    else:
        groups = [np.arange(len(targets))]
    train, val = [], []
    for group in groups:
        shuffled = rng.permutation(group)
        cut = int(round(len(shuffled) * val_fraction))
        val.append(shuffled[:cut])
        train.append(shuffled[cut:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(val))


def build_data(spec: ExperimentSpec, adapter: VisionAdapter, root: Path,
               input_size: tuple[int, int]) -> DataBundle:
    from torch.utils.data import Subset

    train_set = adapter.load(root, "train", _transforms(spec, adapter, input_size, train=True))
    plain = _transforms(spec, adapter, input_size, train=False)
    train_plain = adapter.load(root, "train", plain)
    test_set = adapter.load(root, "test", plain)
    train_idx, val_idx = split_indices(targets_of(train_set), spec.data.split.val_fraction,
                                       stratify=spec.data.split.stratify, seed=spec.seed)
    cap = spec.data.subset
    if cap is not None:
        rng = np.random.default_rng(spec.seed + 1)
        train_idx, val_idx = train_idx[:cap], val_idx[:cap]
        test_set = Subset(test_set, np.sort(rng.permutation(len(test_set))[:cap]).tolist())
    normalization = next((_normalization(step, adapter) for step in spec.data.preprocessing
                          if step.name == "normalize"), None)
    return DataBundle(Subset(train_set, train_idx.tolist()),
                      Subset(train_plain, val_idx.tolist()) if len(val_idx) else None,
                      test_set, adapter.in_channels, input_size, adapter.classes, normalization)
