"""Colour scales between two token colours, interpolated in OKLab so lightness changes evenly."""

from __future__ import annotations

import numpy as np

__all__ = ["colour_ramp"]

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _oklab(hex_colour: str) -> np.ndarray:
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _M2 @ np.cbrt(_M1 @ linear)


def colour_ramp(low: str, high: str, steps: int = 256) -> np.ndarray:
    """(steps × 3) sRGB values 0..1 from ``low`` to ``high`` (``#RRGGBB``)."""
    start, end = _oklab(low), _oklab(high)
    lab = start + np.linspace(0.0, 1.0, steps)[:, None] * (end - start)
    linear = np.clip((np.linalg.inv(_M1) @ ((np.linalg.inv(_M2) @ lab.T) ** 3)).T, 0, 1)
    return np.clip(np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055), 0, 1)
