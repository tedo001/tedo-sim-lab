"""Pictures of a sound: the waveform (min–max per pixel column) and time–frequency images
(spectrogram, mel spectrogram, MFCCs) on one-hue ramps, low frequencies at the bottom."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...theme.ramps import colour_ramp
from ...theme.tokens import COLORS, MONO_FAMILY

__all__ = ["TimeFrequency", "Waveform"]


def _font() -> QFont:
    font = QFont(MONO_FAMILY)
    font.setPixelSize(10)
    return font


class Waveform(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signal = np.zeros(0)
        self.rate = 16_000
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(480, 130)

    def set_signal(self, signal: np.ndarray, rate: int) -> None:
        self.signal, self.rate = np.asarray(signal, dtype=float), rate
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 (Qt)
        painter = QPainter(self)
        area = QRectF(4, 4, self.width() - 8, self.height() - 22)
        painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
        painter.drawLine(QPointF(area.left(), area.center().y()), QPointF(area.right(), area.center().y()))
        if self.signal.size:
            columns = max(int(area.width()), 1)
            edges = np.linspace(0, len(self.signal), columns + 1).astype(int)
            painter.setPen(QPen(QColor(COLORS["series_1"]), 1))
            half = area.height() / 2
            for x in range(columns):
                piece = self.signal[edges[x]:max(edges[x + 1], edges[x] + 1)]
                low, high = float(piece.min()), float(piece.max())
                px = area.left() + x
                painter.drawLine(QPointF(px, area.center().y() - high * half),
                                 QPointF(px, area.center().y() - low * half))
        painter.setPen(QColor(COLORS["text_faint"]))
        painter.setFont(_font())
        seconds = len(self.signal) / self.rate if self.rate else 0
        painter.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 14), Qt.AlignmentFlag.AlignLeft,
                         "0 s")
        painter.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 14),
                         Qt.AlignmentFlag.AlignRight,
                         f"{seconds:.2f} s")
        painter.end()


class TimeFrequency(QWidget):
    """A frames × bins array as an image; ``diverging`` colours around zero (for MFCCs)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.image: QImage | None = None
        self.low_label = self.high_label = ""
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(480, 170)

    def set_values(self, values: np.ndarray, *, low_label: str = "", high_label: str = "",
                   floor: float | None = None,
                   diverging: bool = False) -> None:
        data = np.asarray(values, dtype=float).T[::-1]  # bins upwards, frames to the right
        if data.size == 0:
            self.image = None
            self.update()
            return
        if diverging:
            reach = float(np.abs(data).max()) or 1.0
            negative = colour_ramp(COLORS["heat_zero"], COLORS["heat_neg"])
            positive = colour_ramp(COLORS["heat_zero"], COLORS["heat_pos"])
            t = np.clip(data / reach, -1, 1)
            index = (np.abs(t) * 255).astype(int)
            rgb = np.where((t < 0)[..., None], negative[index], positive[index])
        else:
            top = float(data.max())
            bottom = max(float(data.min()), top - 80.0) if floor is None else floor
            t = np.clip((data - bottom) / ((top - bottom) or 1.0), 0, 1)
            rgb = colour_ramp(COLORS["seq_low"], COLORS["seq_high"])[(t * 255).astype(int)]
        pixels = np.ascontiguousarray((rgb * 255).round().astype(np.uint8))
        self.image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                            QImage.Format.Format_RGB888).copy()
        self.low_label, self.high_label = low_label, high_label
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 (Qt)
        painter = QPainter(self)
        painter.setFont(_font())
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignLeft, self.title)
        area = QRectF(48, 18, self.width() - 52, self.height() - 22)
        if self.image is None:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No sound yet")
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(area, self.image)
        painter.setPen(QColor(COLORS["text_faint"]))
        painter.drawText(QRectF(0, area.top(), 44, 14), Qt.AlignmentFlag.AlignRight, self.high_label)
        painter.drawText(QRectF(0, area.bottom() - 14, 44, 14), Qt.AlignmentFlag.AlignRight, self.low_label)
        painter.end()
