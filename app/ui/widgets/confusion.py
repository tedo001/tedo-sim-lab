"""A confusion matrix: true class down the side, predicted class along the top, one
sequential blue scale (darker = fewer), the count in every cell, the share on hover."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["ConfusionMatrixView", "sequential_colour"]


def _lab(hex_colour: str) -> np.ndarray:
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    return np.cbrt(m1 @ linear), m1


def sequential_colour(fraction: float) -> QColor:
    """0 → ``seq_low``, 1 → ``seq_high``, interpolated in LMS-cube space (OKLab's basis)."""
    low, m1 = _lab(COLORS["seq_low"])
    high, _ = _lab(COLORS["seq_high"])
    mix = (low + min(max(fraction, 0.0), 1.0) * (high - low)) ** 3
    linear = np.clip(np.linalg.inv(m1) @ mix, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    red, green, blue = (int(round(v * 255)) for v in np.clip(srgb, 0, 1))
    return QColor(red, green, blue)


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


class ConfusionMatrixView(QWidget):
    LEFT, TOP = 92, 22

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.matrix = np.zeros((0, 0))
        self.classes: tuple[str, ...] = ()
        self.normalised = False
        self._hover: tuple[int, int] | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_matrix(self, matrix: Sequence[Sequence[float]], classes: Sequence[str]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.classes = tuple(classes)
        self._fit()
        self.update()

    def set_normalised(self, normalised: bool) -> None:
        self.normalised = normalised
        self.update()

    def _cell(self) -> float:
        count = max(len(self.classes), 1)
        return max(min((self.width() - self.LEFT - 4) / count, 46.0), 16.0)

    def _fit(self) -> None:
        self.setFixedHeight(int(self.TOP + self._cell() * max(len(self.classes), 1) + 22))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._fit()
        super().resizeEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(420, self.height())

    def shares(self) -> np.ndarray:
        totals = self.matrix.sum(axis=1, keepdims=True)
        return np.divide(self.matrix, totals, out=np.zeros_like(self.matrix), where=totals > 0)

    def hover_text(self) -> str | None:
        if self._hover is None:
            return None
        true, predicted = self._hover
        count = int(self.matrix[true, predicted])
        share = self.shares()[true, predicted]
        return (f"true {self.classes[true]} → predicted {self.classes[predicted]}: {count} "
                f"({share:.1%} of {self.classes[true]})")

    def paintEvent(self, event: QPaintEvent) -> None:
        if not len(self.classes):
            return
        painter = QPainter(self)
        cell = self._cell()
        values = self.shares() if self.normalised else self.matrix
        peak = values.max() or 1.0
        label_font = _font(FONT_FAMILY, 11)
        metrics = QFontMetrics(label_font)
        for i, name in enumerate(self.classes):
            painter.setFont(label_font)
            painter.setPen(QColor(COLORS["text_dim"]))
            text = metrics.elidedText(name, Qt.TextElideMode.ElideRight, self.LEFT - 8)
            painter.drawText(QRectF(0, self.TOP + i * cell, self.LEFT - 8, cell),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), text)
            top_text = metrics.elidedText(name, Qt.TextElideMode.ElideRight, int(cell))
            painter.drawText(QRectF(self.LEFT + i * cell, 0, cell, self.TOP - 4),
                             int(Qt.AlignmentFlag.AlignCenter), top_text)
        painter.setFont(_font(MONO_FAMILY, 10 if cell >= 30 else 9))
        for true in range(len(self.classes)):
            for predicted in range(len(self.classes)):
                rect = QRectF(self.LEFT + predicted * cell, self.TOP + true * cell, cell, cell)
                fill = sequential_colour(values[true, predicted] / peak)
                painter.fillRect(rect.adjusted(1, 1, -1, -1), fill)
                if cell >= 22:
                    value = values[true, predicted]
                    text = f"{value:.0%}" if self.normalised else f"{int(value)}"
                    painter.setPen(QColor("#0F1114" if fill.lightnessF() > 0.55 else COLORS["text"]))
                    painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        if self._hover is not None:
            true, predicted = self._hover
            painter.setPen(QPen(QColor(COLORS["text"]), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self.LEFT + predicted * cell, self.TOP + true * cell, cell, cell))
        painter.setFont(_font(FONT_FAMILY, 11))
        painter.setPen(QColor(COLORS["text_faint"]))
        bottom = self.TOP + cell * len(self.classes) + 4
        painter.drawText(QRectF(self.LEFT, bottom, cell * len(self.classes), 16),
                         int(Qt.AlignmentFlag.AlignLeft),
                         "rows: true class · columns: predicted class")
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cell = self._cell()
        x, y = event.position().x() - self.LEFT, event.position().y() - self.TOP
        count = len(self.classes)
        hit = (int(y // cell), int(x // cell)) if 0 <= x < cell * count and 0 <= y < cell * count else None
        if hit != self._hover:
            self._hover = hit
            self.setToolTip(self.hover_text() or "")
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = None
        self.update()
