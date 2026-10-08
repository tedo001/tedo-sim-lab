"""Input images for the explainer that need no download.

* Handwritten digits from the UCI "Optical Recognition of Handwritten Digits" set
  (CC BY 4.0), which scikit-learn ships with its code: 8×8 images, scaled up and
  framed the way MNIST frames its digits (a 20×20 digit in a 28×28 image).
* Generated patterns (edges, stripes, shapes, colour fields) that show what
  early kernels respond to.

CNN Explainer's own example images come from Tiny ImageNet, which is derived from
ImageNet; the lab does not copy them.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata

import numpy as np

__all__ = ["DIGITS_CREDIT", "Sample", "digit_samples", "pattern_samples", "prepare_image",
           "resize_bilinear"]

DIGITS_CREDIT = ("UCI Optical Recognition of Handwritten Digits (Alpaydin & Kaynak), CC BY 4.0, "
                 "as shipped with scikit-learn")


@dataclass(frozen=True)
class Sample:
    title: str
    #: (C, H, W), values 0..1.
    image: np.ndarray
    #: Where it came from, for the tooltip.
    credit: str
    #: The true class index, when there is one.
    label: int | None = None


def resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """(H, W) → (height, width), sampling pixel centres like OpenCV and PyTorch."""
    source_h, source_w = image.shape
    ys = np.clip((np.arange(height) + 0.5) * source_h / height - 0.5, 0, source_h - 1)
    xs = np.clip((np.arange(width) + 0.5) * source_w / width - 0.5, 0, source_w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, source_h - 1), np.minimum(x0 + 1, source_w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def prepare_image(rgb: np.ndarray, channels: int, *, invert: bool = False) -> np.ndarray:
    """(H, W, 3) uint8, already at the network's size → (C, H, W) floats in 0..1. One channel
    means luminance (ITU-R 601 weights)."""
    image = np.asarray(rgb, dtype=np.float64) / 255.0
    if channels == 1:
        planes = (image @ np.array([0.299, 0.587, 0.114]))[None]
    else:
        planes = np.moveaxis(image[..., :3], -1, 0)
    return 1.0 - planes if invert else planes


@lru_cache(maxsize=1)
def _uci_digits() -> tuple[np.ndarray, np.ndarray] | None:
    """(1797, 8, 8) values 0..16 and their labels, read from scikit-learn's bundled file without
    importing scikit-learn (which takes seconds); falls back to ``load_digits``. ``None`` when
    scikit-learn is not installed."""
    try:
        path = metadata.distribution("scikit-learn").locate_file("sklearn/datasets/data/digits.csv.gz")
        with gzip.open(str(path), "rt") as file:
            table = np.loadtxt(file, delimiter=",")
        return table[:, :-1].reshape(-1, 8, 8), table[:, -1].astype(int)
    except (OSError, metadata.PackageNotFoundError, ValueError):
        try:
            from sklearn.datasets import load_digits
        except ImportError:
            return None
        digits = load_digits()
        return digits.images, digits.target


def digit_samples(size: int = 28, channels: int = 1) -> list[Sample]:
    """The first example of each digit 0–9, framed like MNIST at ``size``; none without
    scikit-learn."""
    found = _uci_digits()
    if found is None:
        return []
    images, labels = found
    inner = max(round(size * 20 / 28), 8)
    offset = (size - inner) // 2
    samples = []
    for digit in range(10):
        small = images[int(np.argmax(labels == digit))] / 16.0
        canvas = np.zeros((size, size))
        canvas[offset:offset + inner, offset:offset + inner] = np.clip(
            resize_bilinear(small, inner, inner), 0.0, 1.0)
        samples.append(Sample(f"Digit {digit}", np.repeat(canvas[None], channels, axis=0),
                              DIGITS_CREDIT, digit))
    return samples


def pattern_samples(size: int, channels: int) -> list[Sample]:
    """Generated test patterns at ``size`` × ``size``."""
    grid = (np.arange(size) + 0.5) / size
    y, x = np.meshgrid(grid, grid, indexing="ij")
    radius = np.hypot(y - 0.5, x - 0.5)
    ring = (np.abs(radius - 0.3) < 0.07).astype(float)
    disc = (radius < 0.3).astype(float)
    patterns = {
        "Vertical edge": (x > 0.5).astype(float),
        "Horizontal stripes": (np.floor(y * 6) % 2).astype(float),
        "Diagonal": (y > x).astype(float),
        "Checkerboard": ((np.floor(y * 4) + np.floor(x * 4)) % 2).astype(float),
        "Ring": ring,
        "Cross": ((np.abs(y - 0.5) < 0.08) | (np.abs(x - 0.5) < 0.08)).astype(float),
    }
    samples = [Sample(title, np.repeat(plane[None], channels, axis=0), "Generated")
               for title, plane in patterns.items()]
    if channels == 3:
        samples.append(Sample("Red disc on blue", np.stack([disc, np.zeros_like(disc), 1 - disc]),
                              "Generated"))
        samples.append(Sample("Colour gradient", np.stack([x, y, 1 - x]), "Generated"))
    return samples
