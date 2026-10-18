"""Test images drawn with numpy (no files, no licences): shapes on a gradient, a checkerboard, a
zone plate (rings whose frequency rises outwards — aliasing and filters show at once), noisy
steps, and a scene of text-like strokes."""

from __future__ import annotations

import numpy as np

__all__ = ["SAMPLES", "sample"]


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return x / (size - 1), y / (size - 1)


def shapes(size: int = 256) -> np.ndarray:
    x, y = _grid(size)
    image = np.stack([0.15 + 0.25 * x, 0.18 + 0.2 * y, 0.28 + 0.1 * (1 - x)], axis=-1)
    disc = (x - 0.3) ** 2 + (y - 0.32) ** 2 < 0.15 ** 2
    square = (np.abs(x - 0.7) < 0.14) & (np.abs(y - 0.3) < 0.14)
    triangle = (y > 0.55) & (y < 0.9) & (np.abs(x - 0.5) < (y - 0.55) * 0.6)
    image[disc] = (0.85, 0.35, 0.25)
    image[square] = (0.25, 0.55, 0.85)
    image[triangle] = (0.35, 0.75, 0.45)
    return image.astype(np.float32)


def checkerboard(size: int = 256, squares: int = 8) -> np.ndarray:
    x, y = _grid(size)
    cells = np.floor(x * squares * 0.999) + np.floor(y * squares * 0.999)
    return ((cells % 2) * 0.8 + 0.1).astype(np.float32)


def zone_plate(size: int = 256) -> np.ndarray:
    x, y = _grid(size)
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    return (0.5 + 0.5 * np.cos(np.pi * size * 0.9 * r2)).astype(np.float32)


def noisy_steps(size: int = 256, seed: int = 7) -> np.ndarray:
    x, _y = _grid(size)
    steps = np.floor(x * 5) / 4
    return np.clip(steps + np.random.default_rng(seed).normal(0, 0.06, steps.shape), 0, 1).astype(np.float32)


def strokes(size: int = 256, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 0.92, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    height = max(2, int(size * 0.055))
    for row in range(6):
        top = int(size * (0.09 + row * 0.15))
        for _ in range(rng.integers(4, 9)):
            left = int(rng.integers(int(size * 0.04), max(int(size * 0.04) + 1, int(size * 0.84))))
            width = int(rng.integers(max(2, int(size * 0.03)), max(3, int(size * 0.14))))
            image[(yy >= top) & (yy < top + height) & (xx >= left) & (xx < left + width)] = 0.12
    return image


SAMPLES = {"Shapes": shapes, "Checkerboard": checkerboard, "Zone plate": zone_plate,
           "Noisy steps": noisy_steps, "Text-like strokes": strokes}


def sample(name: str, size: int = 256) -> np.ndarray:
    return SAMPLES[name](size)
