"""Parallel coordinates for a sweep: one vertical axis per swept parameter and a last one for the
score; each run is a line through its values, coloured by its score on one sequential blue scale
(darker = lower, lighter = higher), so the settings that lead to good runs stand out. Numeric axes
run from their minimum (bottom) to maximum (top); text values sit evenly spaced. A missing value
leaves a gap. Hover a line to read the run's values; drawn with QPainter, no OpenGL."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.ramps import colour_ramp
from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["Axis", "ParallelCoordinates"]

_RAMP = colour_ramp(COLORS["surface_low"], COLORS["seq_high"], 64)


@dataclass
class Axis:
    name: str
    values: list[Any]

    @property
    def numeric(self) -> bool:
        return all(v is None or (isinstance(v, int | float) and not isinstance(v, bool)) for v in self.values)

    def levels(self) -> list[str]:
        return sorted({str(v) for v in self.values if v is not None})

    def position(self, value: Any) -> float | None:
        """0 (bottom) … 1 (top), or None for a missing value."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if self.numeric:
            numbers = [float(v) for v in self.values if v is not None]
            low, high = min(numbers), max(numbers)
            return 0.5 if high == low else (float(value) - low) / (high - low)
        levels = self.levels()
        return 0.5 if len(levels) == 1 else levels.index(str(value)) / (len(levels) - 1)

    def ticks(self) -> list[tuple[float, str]]:
        if self.numeric:
            numbers = [float(v) for v in self.values if v is not None]
            if not numbers:
                return []
            low, high = min(numbers), max(numbers)
            return [(0.0, _short(low)), (1.0, _short(high))] if high != low else [(0.5, _short(low))]
        levels = self.levels()
        return [(self.position(level) or 0.0, level) for level in levels]


def _short(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e4 else f"{value:.1e}"


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


class ParallelCoordinates(QWidget):
    TOP, BOTTOM, SIDE = 40, 26, 34

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.axes: list[Axis] = []
        self.labels: list[str] = []
        self._mouse: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(520, 300)

    def set_data(self, axes: Sequence[Axis], labels: Sequence[str]) -> None:
        """``axes``: the parameters, then the score (it colours the lines); ``labels``: one per run."""
        self.axes, self.labels = list(axes), list(labels)
        self.update()

    def _x(self, index: int) -> float:
        span = max(self.width() - 2 * self.SIDE, 1)
        return self.SIDE + (span * index / (len(self.axes) - 1) if len(self.axes) > 1 else span / 2)

    def _y(self, position: float) -> float:
        return self.height() - self.BOTTOM - position * max(self.height() - self.TOP - self.BOTTOM, 1)

    def _colour(self, run: int) -> QColor:
        position = self.axes[-1].position(self.axes[-1].values[run])
        if position is None:
            return QColor(COLORS["series_other"])
        r, g, b = _RAMP[min(int(position * (len(_RAMP) - 1)), len(_RAMP) - 1)]
        return QColor.fromRgbF(float(r), float(g), float(b))

    def _points(self, run: int) -> list[QPointF | None]:
        out = []
        for index, axis in enumerate(self.axes):
            position = axis.position(axis.values[run])
            out.append(None if position is None else QPointF(self._x(index), self._y(position)))
        return out

    def hovered(self) -> int | None:
        """The run whose line passes closest to the mouse (within 8 px)."""
        if self._mouse is None or len(self.axes) < 2:
            return None
        best, found = 8.0, None
        for run in range(len(self.labels)):
            points = self._points(run)
            for a, b in zip(points, points[1:], strict=False):
                if a is None or b is None or not (a.x() <= self._mouse.x() <= b.x()):
                    continue
                t = (self._mouse.x() - a.x()) / ((b.x() - a.x()) or 1)
                distance = abs(a.y() + t * (b.y() - a.y()) - self._mouse.y())
                if distance < best:
                    best, found = distance, run
        return found

    def hover_text(self) -> str | None:
        run = self.hovered()
        if run is None:
            return None
        parts = [f"{axis.name} {_short(v) if isinstance(v, float) else v}"
                 for axis in self.axes if (v := axis.values[run]) is not None]
        return f"{self.labels[run]}: " + " · ".join(parts)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if len(self.axes) < 2 or not self.labels:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.setFont(_font(FONT_FAMILY, 12))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "A finished sweep draws here")
            return
        top, bottom = self._y(1.0), self._y(0.0)
        for index, axis in enumerate(self.axes):  # recessive axes with their names and ticks
            x = self._x(index)
            painter.setPen(QPen(QColor(COLORS["border"]), 1))
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            painter.setFont(_font(FONT_FAMILY, 11))
            painter.setPen(QColor(COLORS["text_dim"] if index < len(self.axes) - 1 else COLORS["text"]))
            width = max(self.width() / len(self.axes) - 4, 40)
            name = painter.fontMetrics().elidedText(axis.name, Qt.TextElideMode.ElideLeft, int(width))
            painter.drawText(QRectF(x - width / 2, 4, width, 16), int(Qt.AlignmentFlag.AlignHCenter), name)
            painter.setFont(_font(MONO_FAMILY, 10))
            painter.setPen(QColor(COLORS["text_faint"]))
            for position, text in axis.ticks()[:8]:
                painter.drawText(QRectF(x + 4, self._y(position) - 7, 70, 14),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
        hovered = self.hovered()
        order = sorted(range(len(self.labels)), key=lambda r: self.axes[-1].position(self.axes[-1].values[r])
                       or -1.0)  # best runs on top
        for run in order:
            colour = self._colour(run)
            if hovered is not None and run != hovered:
                colour.setAlphaF(0.35)
            painter.setPen(QPen(colour, 3 if run == hovered else 2))
            points = self._points(run)
            for a, b in zip(points, points[1:], strict=False):
                if a is not None and b is not None:
                    painter.drawLine(a, b)
        painter.setFont(_font(MONO_FAMILY, 11))
        painter.setPen(QColor(COLORS["text_dim"]))
        text = self.hover_text() or f"colour: {self.axes[-1].name} (darker lower, lighter higher)"
        painter.drawText(QRectF(self.SIDE, self.height() - 18, self.width() - 2 * self.SIDE, 16),
                         int(Qt.AlignmentFlag.AlignLeft), text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse = event.position()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802, ANN001 (Qt override)
        self._mouse = None
        self.update()
