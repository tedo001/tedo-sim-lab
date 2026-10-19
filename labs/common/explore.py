"""A dataset as points to look at in 3-D: a table's numeric columns, or an MNIST-style image set's
pixels, with each row's class (or a number to colour by), and principal components for data with
more columns than three.

Image sets are read straight from their IDX files (the format MNIST and Fashion-MNIST ship in),
so this needs neither PyTorch nor a pickle. Principal components come from a singular value
decomposition of the centred columns (scaled to unit variance for tables, whose columns have
different units; left as they are for pixels, which share one).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_registry.adapter import DatasetAdapter, TabularAdapter

__all__ = ["IDX_SETS", "PointTable", "explorable", "image_points", "principal_components", "read_idx",
           "tabular_points"]

#: Image sets read from IDX files: adapter class name → the folder holding the test files.
IDX_SETS = {"MnistAdapter": "MNIST/raw", "FashionMnistAdapter": "FashionMNIST/raw"}
GROUP_LIMIT = 20  # a target with more distinct values than this is a number to colour by


@dataclass
class PointTable:
    name: str
    #: The columns the points can be drawn along (for images: none, only components).
    columns: list[str]
    values: np.ndarray
    groups: np.ndarray | None = None
    group_names: list[str] = field(default_factory=list)
    #: A number per row to colour by (a numeric target with many values).
    numbers: np.ndarray | None = None
    number_name: str = ""
    #: Rows in the dataset (before sampling) and what was left out, for the caption.
    total: int = 0
    note: str = ""
    #: Whether components should be computed on unit-variance columns.
    standardize: bool = True

    @property
    def rows(self) -> int:
        return len(self.values)


def explorable(adapter: DatasetAdapter) -> bool:
    """Tables and MNIST-style image sets can be drawn as points."""
    return isinstance(adapter, TabularAdapter) or type(adapter).__name__ in IDX_SETS


def _sample(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))


def tabular_points(adapter: TabularAdapter, root: Path, *, limit: int = 5000, seed: int = 0) -> PointTable:
    import pandas as pd

    features, target = adapter.load_frame(root)
    numeric = features.select_dtypes(include=["number", "bool"])
    left_out = [str(c) for c in features.columns if c not in numeric.columns]
    rows = _sample(len(numeric), limit, seed)
    values = numeric.iloc[rows].to_numpy(dtype=float)
    table = PointTable(adapter.card.name, [str(c) for c in numeric.columns], values, total=len(numeric))
    if target is not None:
        target = target.iloc[rows]
        known = target.notna().to_numpy()
        if not pd.api.types.is_numeric_dtype(target) or target.nunique() <= GROUP_LIMIT:
            categories = pd.Categorical(target)
            labels = list(categories.categories)
            names = getattr(adapter, "class_names", lambda: ())()
            if names and len(names) == len(labels) and labels == list(range(len(labels))):
                table.group_names = list(names)
            else:
                table.group_names = [str(label) for label in labels]
            table.groups = np.asarray(categories.codes, dtype=int)
            table.values, table.groups = table.values[known], table.groups[known]
        else:
            table.values, table.numbers = table.values[known], target.to_numpy(dtype=float)[known]
            table.number_name = str(target.name)
    parts = []
    if left_out:
        parts.append(f"{len(left_out)} text column{'s' if len(left_out) != 1 else ''} left out "
                     f"({', '.join(left_out[:3])}{'…' if len(left_out) > 3 else ''})")
    if table.total > len(rows):
        parts.append(f"{len(rows):,} of {table.total:,} rows, sampled")
    table.note = "; ".join(parts)
    return table


def read_idx(path: Path) -> np.ndarray:
    """An IDX file (optionally gzipped): a magic number naming the type and dimensions, then data."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path} is not an IDX file")
    kinds = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}
    if raw[2] not in kinds:
        raise ValueError(f"{path} holds an unknown IDX type {raw[2]:#x}")
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    data = np.frombuffer(raw, dtype=kinds[raw[2]], offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path} is cut short")
    return data.reshape(shape)


def _file(folder: Path, stem: str) -> Path:
    for candidate in (folder / stem, folder / f"{stem}.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{folder / stem} is missing: download the dataset in the Dataset Hub")


def image_points(adapter: DatasetAdapter, root: Path, *, limit: int = 3000, seed: int = 0) -> PointTable:
    """The test images of an MNIST-style set as rows of pixels (0..1)."""
    folder = adapter.data_dir(root) / IDX_SETS[type(adapter).__name__]
    images = read_idx(_file(folder, "t10k-images-idx3-ubyte"))
    labels = read_idx(_file(folder, "t10k-labels-idx1-ubyte")).astype(int)
    rows = _sample(len(images), limit, seed)
    classes: Any = getattr(adapter, "classes", ())
    names = list(classes) if len(classes) else [str(i) for i in range(int(labels.max()) + 1)]
    return PointTable(adapter.card.name, [], images[rows].reshape(len(rows), -1).astype(float) / 255.0,
                      labels[rows], names, total=len(images), standardize=False,
                      note=f"{len(rows):,} of the {len(images):,} test images, sampled; each is "
                           f"{images.shape[1]} × {images.shape[2]} = "
                           f"{images.shape[1] * images.shape[2]} pixels")


def principal_components(values: np.ndarray, count: int = 3, *,
                         standardize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(rows × count scores, the share of the variance each component holds, all of them)."""
    data = np.asarray(values, dtype=float)
    mean = np.nanmean(data, axis=0)
    data = np.where(np.isnan(data), mean, data) - mean
    if standardize:
        spread = data.std(axis=0)
        data = data / np.where(spread > 0, spread, 1.0)
    _u, singular, vt = np.linalg.svd(data, full_matrices=False)
    variance = singular ** 2
    share = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    scores = data @ vt[:count].T
    if scores.shape[1] < count:
        scores = np.pad(scores, ((0, 0), (0, count - scores.shape[1])))
    return scores, share
