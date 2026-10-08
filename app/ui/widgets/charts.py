"""Small, live charts drawn with QPainter, following the lab's data-viz rules.

One series per chart (small multiples, so no legend and nothing told apart by
colour alone), a single 0..maximum axis, 2 px lines, recessive grid, text in
text colours, and a hover crosshair with the exact value. Missing values leave
a gap rather than a fake zero.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, MONO_FAMILY

__all__ = ["MeterBar", "TimeSeriesChart"]

Point = tuple[float, float | None]


def _mono(pixel_size: int) -> QFont:
    font = QFont(MONO_FAMILY)
    font.setPixelSize(pixel_size)
    return font


def _ago(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return "now" if seconds < 2 else f"{seconds} s ago" if seconds < 90 else f"{seconds // 60} min ago"


class TimeSeriesChart(QWidget):
    """The last ``window_s`` seconds of one percentage-like series."""

    LEFT, RIGHT, TOP, BOTTOM = 30, 8, 8, 18

    def __init__(self, *, maximum: float = 100.0, unit: str = "%", window_s: float = 120.0,
                 colour: str = COLORS["series_1"], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.maximum, self.unit, self.window_s = maximum, unit, window_s
        self.colour = QColor(colour)
        self._points: list[Point] = []
        self._empty_text = "Waiting for data…"
        self._hover_x: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(320, 130)

    def set_points(self, points: Sequence[Point]) -> None:
        self._points = list(points)
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    def points(self) -> list[Point]:
        return list(self._points)

    # Geometry ---------------------------------------------------------------
    def _plot(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(self.width() - self.LEFT - self.RIGHT, 1),
                      max(self.height() - self.TOP - self.BOTTOM, 1))

    def _xy(self, plot: QRectF, now: float, ts: float, value: float) -> QPointF:
        x = plot.right() - (now - ts) / self.window_s * plot.width()
        y = plot.bottom() - min(max(value / self.maximum, 0.0), 1.0) * plot.height()
        return QPointF(x, y)

    def _visible(self, now: float) -> list[Point]:
        return [(ts, value) for ts, value in self._points if now - ts <= self.window_s]

    # Painting ---------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = self._plot()
        now = self._points[-1][0] if self._points else time.time()
        visible = self._visible(now)
        if not any(value is not None for _, value in visible):  # say why, without a fake axis
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self._empty_text)
            return
        self._paint_axes(painter, plot)
        self._paint_series(painter, plot, now, visible)
        if self._hover_x is not None:
            self._paint_hover(painter, plot, now, visible)

    def _paint_axes(self, painter: QPainter, plot: QRectF) -> None:
        painter.setFont(_mono(10))
        for fraction in (0.0, 0.5, 1.0):
            y = plot.bottom() - fraction * plot.height()
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(QRectF(0, y - 7, self.LEFT - 6, 14),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{fraction * self.maximum:g}")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 3, 120, 14),
                         Qt.AlignmentFlag.AlignLeft, _ago(self.window_s))
        painter.drawText(QRectF(plot.right() - 60, plot.bottom() + 3, 60, 14),
                         Qt.AlignmentFlag.AlignRight, "now")

    def _segments(self, plot: QRectF, now: float, visible: list[Point]) -> list[list[QPointF]]:
        segments: list[list[QPointF]] = [[]]
        for ts, value in visible:
            if value is None:
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(self._xy(plot, now, ts, value))
        return [segment for segment in segments if segment]

    def _paint_series(self, painter: QPainter, plot: QRectF, now: float,
                      visible: list[Point]) -> None:
        fill = QColor(self.colour)
        fill.setAlphaF(0.14)
        for segment in self._segments(plot, now, visible):
            line = QPainterPath(segment[0])
            for point in segment[1:]:
                line.lineTo(point)
            area = QPainterPath(line)
            area.lineTo(segment[-1].x(), plot.bottom())
            area.lineTo(segment[0].x(), plot.bottom())
            area.closeSubpath()
            painter.fillPath(area, fill)
            pen = QPen(self.colour, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(line, pen)

    def _nearest(self, plot: QRectF, now: float, visible: list[Point]) -> tuple[float, float] | None:
        """The sampled point closest to the mouse."""
        candidates = [(ts, value) for ts, value in visible if value is not None]
        if self._hover_x is None or not candidates:
            return None
        return min(candidates, key=lambda p: abs(self._xy(plot, now, p[0], p[1]).x() - self._hover_x))

    def _paint_hover(self, painter: QPainter, plot: QRectF, now: float,
                     visible: list[Point]) -> None:
        nearest = self._nearest(plot, now, visible)
        if nearest is None:
            return
        ts, value = nearest
        point = self._xy(plot, now, ts, value)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawLine(QPointF(point.x(), plot.top()), QPointF(point.x(), plot.bottom()))
        painter.setPen(QPen(QColor(COLORS["surface"]), 2))  # 2 px surface ring
        painter.setBrush(self.colour)
        painter.drawEllipse(point, 4.5, 4.5)
        text = f"{value:.0f}{self.unit} · {_ago(now - ts)}"
        painter.setFont(_mono(11))
        width = painter.fontMetrics().horizontalAdvance(text) + 14
        box = QRectF(min(point.x() + 8, plot.right() - width), plot.top() + 2, width, 20)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def hover_text(self) -> str | None:
        """What the tooltip would say (tests and accessibility)."""
        if not self._points:
            return None
        now = self._points[-1][0]
        nearest = self._nearest(self._plot(), now, self._visible(now))
        if nearest is None:
            return None
        ts, value = nearest
        return f"{value:.0f}{self.unit} · {_ago(now - ts)}"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover_x = event.position().x()
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_x = None
        self.update()


class MeterBar(QWidget):
    """``CPU ▬▬▬▭▭ 28%``: a compact live gauge for the title row."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self._fraction: float | None = None
        self._text = "—"
        self.setFixedSize(QSize(104, 20))

    def set_value(self, fraction: float | None, text: str, tooltip: str = "") -> None:
        self._fraction = None if fraction is None else min(max(fraction, 0.0), 1.0)
        self._text = text
        self.setToolTip(tooltip)
        self.setAccessibleName(f"{self.label} {text}")
        self.update()

    def text(self) -> str:
        return self._text

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_mono(10))
        painter.setPen(QColor(COLORS["text_faint"]))
        painter.drawText(QRectF(0, 0, 32, 20), Qt.AlignmentFlag.AlignVCenter, self.label)
        track = QRectF(34, 8, 36, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["border"]))
        painter.drawRoundedRect(track, 2, 2)
        if self._fraction:
            painter.setBrush(QColor(COLORS["series_1"]))
            painter.drawRoundedRect(QRectF(track.left(), track.top(),
                                           max(track.width() * self._fraction, 3), 4), 2, 2)
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(76, 0, 28, 20),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text)
