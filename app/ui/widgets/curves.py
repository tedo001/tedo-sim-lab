"""A training curve: one metric over epochs, drawn with QPainter by the lab's chart rules
(one series, one axis, 2 px line, markers while there are few points, recessive grid,
hover snaps to the nearest epoch and shows its value)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["EpochChart"]


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


def _nice(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3g}" if abs(value) >= 1e-3 else f"{value:.1e}"


class EpochChart(QWidget):
    """``points`` are (epoch, value). ``percent`` draws a 0–1 metric on a fixed 0..1 axis."""

    LEFT, RIGHT, TOP, BOTTOM = 44, 10, 26, 20

    def __init__(self, title: str, *, percent: bool = False, colour: str = COLORS["series_1"],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title, self.percent, self.colour = title, percent, QColor(colour)
        self._points: list[tuple[int, float]] = []
        self._epochs = 0
        self._hover_x: float | None = None
        self._empty = "Appears after the first epoch"
        self.setMouseTracking(True)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(300, 170)

    def set_points(self, points: Sequence[tuple[int, float]], epochs: int | None = None) -> None:
        self._points = sorted((int(e), float(v)) for e, v in points if v is not None and math.isfinite(v))
        self._epochs = max(epochs or 0, self._points[-1][0] if self._points else 0)
        self.update()

    def points(self) -> list[tuple[int, float]]:
        return list(self._points)

    # Geometry -------------------------------------------------------------------
    def _range(self) -> tuple[float, float]:
        if self.percent:
            return 0.0, 1.0
        values = [value for _, value in self._points]
        low, high = min(values + [0.0]), max(values)
        return low, high if high > low else low + 1.0

    def _plot(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(self.width() - self.LEFT - self.RIGHT, 1),
                      max(self.height() - self.TOP - self.BOTTOM, 1))

    def _xy(self, plot: QRectF, epoch: int, value: float) -> QPointF:
        low, high = self._range()
        span = max(self._epochs - 1, 1)
        x = plot.left() + (epoch - 1) / span * plot.width() if self._epochs > 1 else plot.center().x()
        return QPointF(x, plot.bottom() - (value - low) / (high - low) * plot.height())

    def nearest(self) -> tuple[int, float] | None:
        if self._hover_x is None or not self._points:
            return None
        plot = self._plot()
        return min(self._points, key=lambda p: abs(self._xy(plot, *p).x() - self._hover_x))

    def hover_text(self) -> str | None:
        found = self.nearest()
        return None if found is None else f"epoch {found[0]} · {self._format(found[1])}"

    def _format(self, value: float) -> str:
        return f"{value:.2%}" if self.percent else _nice(value)

    # Painting -------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignLeft), self.title)
        if self._points:
            latest = self._format(self._points[-1][1])
            painter.setFont(_font(MONO_FAMILY, 12))
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignRight), latest)
        plot = self._plot()
        if not self._points:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(plot, int(Qt.AlignmentFlag.AlignCenter), self._empty)
            return
        self._paint_axes(painter, plot)
        self._paint_series(painter, plot)
        found = self.nearest()
        if found is not None:
            self._paint_hover(painter, plot, found)

    def _paint_axes(self, painter: QPainter, plot: QRectF) -> None:
        low, high = self._range()
        painter.setFont(_font(MONO_FAMILY, 10))
        for fraction in (0.0, 0.5, 1.0):
            y = plot.bottom() - fraction * plot.height()
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(COLORS["text_faint"]))
            value = low + fraction * (high - low)
            text = f"{value:.0%}" if self.percent else _nice(value)
            painter.drawText(QRectF(0, y - 7, self.LEFT - 6, 14),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), text)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 3, 80, 14), int(Qt.AlignmentFlag.AlignLeft),
                         "epoch 1")
        painter.drawText(QRectF(plot.right() - 80, plot.bottom() + 3, 80, 14),
                         int(Qt.AlignmentFlag.AlignRight), f"{self._epochs}")

    def _paint_series(self, painter: QPainter, plot: QRectF) -> None:
        points = [self._xy(plot, *p) for p in self._points]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        pen = QPen(self.colour, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)
        if len(points) <= 30:  # markers while they stay legible
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))
            painter.setBrush(self.colour)
            for point in points:
                painter.drawEllipse(point, 4, 4)

    def _paint_hover(self, painter: QPainter, plot: QRectF, found: tuple[int, float]) -> None:
        point = self._xy(plot, *found)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawLine(QPointF(point.x(), plot.top()), QPointF(point.x(), plot.bottom()))
        painter.setPen(QPen(QColor(COLORS["surface"]), 2))
        painter.setBrush(self.colour)
        painter.drawEllipse(point, 5, 5)
        text = self.hover_text() or ""
        painter.setFont(_font(MONO_FAMILY, 11))
        width = painter.fontMetrics().horizontalAdvance(text) + 14
        box = QRectF(min(max(point.x() + 8, plot.left()), plot.right() - width), plot.top() + 2, width, 20)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover_x = event.position().x()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover_x = None
        self.update()
