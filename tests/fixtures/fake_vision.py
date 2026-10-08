"""A tiny MNIST-format dataset written to disk, so the training runner can be tested
without downloading anything. Class ``k`` is a bright square at a position that depends
on ``k``, plus noise: easy enough that a few steps of training learn it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["make_images", "write_fake_mnist"]


def make_images(count: int, seed: int, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 10
    images = rng.integers(0, 40, size=(count, size, size), dtype=np.uint8)
    for index, label in enumerate(labels):
        top, left = 2 + (label // 5) * 12, 2 + (label % 5) * 5
        images[index, top:top + 10, left:left + 4] = 230
    return images, labels.astype(np.uint8)


def _idx(path: Path, array: np.ndarray, kind: int) -> None:
    header = np.array([0, 0, 8, array.ndim], dtype=np.uint8).tobytes()
    dims = np.array(array.shape, dtype=">i4").tobytes()
    path.write_bytes(header + dims + array.astype(np.uint8).tobytes())


def write_fake_mnist(datasets_root: Path, *, card_id: str = "mnist", folder: str = "MNIST",
                     train: int = 200, test: int = 60) -> Path:
    raw = datasets_root / card_id / folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for prefix, count, seed in (("train", train, 1), ("t10k", test, 2)):
        images, labels = make_images(count, seed)
        _idx(raw / f"{prefix}-images-idx3-ubyte", images, 3)
        _idx(raw / f"{prefix}-labels-idx1-ubyte", labels, 1)
    return raw
