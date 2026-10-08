"""Colour scales for activation maps, kernels and inputs, rendered to ``QImage``.

Activations and weights use one diverging scale centred on zero (CNN Explainer uses
red–blue too): red below zero, blue above, a neutral grey at zero, with lightness
rising with magnitude. Input channels are plain greys, 0 black → 1 white.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen

from ...theme.tokens import COLORS, MONO_FAMILY

__all__ = ["diverging_lut", "grey_image", "heat_colour", "heat_image", "lightness", "mono_font",
           "paint_legend", "rgb_image"]

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _to_oklab(hex_colour: str) -> np.ndarray:
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _M2 @ np.cbrt(_M1 @ linear)


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    """(N, 3) OKLab → (N, 3) sRGB bytes."""
    lms = (np.linalg.inv(_M2) @ lab.T) ** 3
    linear = np.clip(np.linalg.inv(_M1) @ lms, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.round(np.clip(srgb.T, 0, 1) * 255).astype(np.uint8)


def lightness(hex_colour: str) -> float:
    """OKLab lightness, 0..1."""
    return float(_to_oklab(hex_colour)[0])


@lru_cache(maxsize=1)
def diverging_lut() -> np.ndarray:
    """(256,) ARGB32 values: index 0 = −limit, 128 ≈ 0, 255 = +limit."""
    zero, negative, positive = (_to_oklab(COLORS[k]) for k in ("heat_zero", "heat_neg", "heat_pos"))
    t = np.linspace(-1, 1, 256)[:, None]
    lab = np.where(t < 0, zero + (-t) * (negative - zero), zero + t * (positive - zero))
    rgb = _from_oklab(lab).astype(np.uint32)
    return (0xFF000000 | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32)


def _image(argb: np.ndarray) -> QImage:
    argb = np.ascontiguousarray(argb, dtype=np.uint32)
    height, width = argb.shape
    image = QImage(argb.data, width, height, width * 4, QImage.Format.Format_ARGB32)
    return image.copy()  # own the pixels; the numpy buffer goes away


def _indices(values: np.ndarray, limit: float) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float64) / max(limit, 1e-12) + 1) / 2
    return np.clip(np.round(scaled * 255), 0, 255).astype(np.intp)


def heat_image(values: np.ndarray, limit: float) -> QImage:
    """A 2-D array on the diverging scale spanning −limit … +limit, one pixel per value."""
    return _image(diverging_lut()[_indices(values, limit)])


def heat_colour(value: float, limit: float) -> QColor:
    return QColor.fromRgba(int(diverging_lut()[_indices(np.array([value]), limit)[0]]))


def grey_image(values: np.ndarray) -> QImage:
    level = np.clip(np.round(np.asarray(values) * 255), 0, 255).astype(np.uint32)
    return _image(0xFF000000 | (level << 16) | (level << 8) | level)


def rgb_image(chw: np.ndarray) -> QImage:
    """(3, H, W) or (1, H, W) values 0..1 → the image as a person sees it."""
    planes = np.clip(np.round(np.asarray(chw) * 255), 0, 255).astype(np.uint32)
    if planes.shape[0] == 1:
        return grey_image(chw[0])
    return _image(0xFF000000 | (planes[0] << 16) | (planes[1] << 8) | planes[2])


def mono_font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(MONO_FAMILY)
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    return font


def paint_legend(painter: QPainter, rect: QRectF, limit: float | None, *, title: str = "") -> None:
    """A gradient bar with −limit, 0, +limit beneath it; ``limit=None`` draws the 0..1 greys."""
    gradient = QLinearGradient(rect.topLeft(), rect.topRight())
    if limit is None:
        gradient.setColorAt(0, QColor("#000000"))
        gradient.setColorAt(1, QColor("#FFFFFF"))
        ticks = ("0", "", "1")
    else:
        lut = diverging_lut()
        for stop in range(0, 256, 32):
            gradient.setColorAt(stop / 255, QColor.fromRgba(int(lut[stop])))
        gradient.setColorAt(1, QColor.fromRgba(int(lut[255])))
        ticks = (f"{-limit:.3g}", "0", f"{limit:.3g}")
    bar = QRectF(rect.left(), rect.top() + 14, rect.width(), 8)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRect(bar)
    painter.setPen(QPen(QColor(COLORS["text_faint"])))
    painter.setFont(mono_font(10))
    if title:
        painter.drawText(QRectF(rect.left(), rect.top(), rect.width(), 13),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), title)
    below = QRectF(rect.left(), bar.bottom() + 2, rect.width(), 13)
    for text, align in zip(ticks, (Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignHCenter,
                                    Qt.AlignmentFlag.AlignRight), strict=True):
        painter.drawText(below, int(align | Qt.AlignmentFlag.AlignTop), text)
