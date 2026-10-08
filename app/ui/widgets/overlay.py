"""One metric for several runs on one axis (Compare Experiments): up to eight series in the
fixed categorical order, 2 px lines, a legend always, direct labels at the line ends when there
are four or fewer, and a crosshair that lists every run's value at the nearest epoch. Text is
drawn in text colours; the colour swatch beside it carries the identity."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY, SERIES

__all__ = ["OverlayChart"]

Series = tuple[str, Sequence[tuple[int, float]]]


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


class OverlayChart(QWidget):
    LEFT, RIGHT, TOP, BOTTOM = 48, 110, 26, 44

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = ""
        self.percent = False
        self.series: list[Series] = []
        self._hover_x: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(520, 300)

    def set_series(self, title: str, series: Sequence[Series], *, percent: bool = False) -> None:
        if len(series) > len(SERIES):
            raise ValueError(f"at most {len(SERIES)} series: colours are never repeated")
        self.title, self.percent = title, percent
        self.series = [(name, sorted((int(e), float(v)) for e, v in points
                                     if v is not None and math.isfinite(v))) for name, points in series]
        self.update()

    # Geometry -------------------------------------------------------------------------
    def _right(self) -> int:
        return self.RIGHT if len(self.series) <= 4 else 12

    def _area(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(self.width() - self.LEFT - self._right(), 1),
                      max(self.height() - self.TOP - self.BOTTOM, 1))

    def _ranges(self) -> tuple[int, float, float]:
        epochs = max((p[-1][0] for _, p in self.series if p), default=1)
        values = [v for _, points in self.series for _, v in points]
        if self.percent:
            return epochs, 0.0, 1.0
        low, high = min(values + [0.0]), max(values or [1.0])
        return epochs, low, high if high > low else low + 1.0

    def _xy(self, area: QRectF, epoch: int, value: float) -> QPointF:
        epochs, low, high = self._ranges()
        x = area.left() + (epoch - 1) / max(epochs - 1, 1) * area.width() if epochs > 1 else area.center().x()
        return QPointF(x, area.bottom() - (value - low) / (high - low) * area.height())

    def nearest_epoch(self) -> int | None:
        if self._hover_x is None or not any(points for _, points in self.series):
            return None
        epochs, _low, _high = self._ranges()
        area = self._area()
        fraction = (self._hover_x - area.left()) / max(area.width(), 1)
        return min(max(round(fraction * (epochs - 1)) + 1, 1), epochs)

    def hover_lines(self) -> list[str]:
        epoch = self.nearest_epoch()
        if epoch is None:
            return []
        lines = [f"epoch {epoch}"]
        for name, points in self.series:
            value = dict(points).get(epoch)
            lines.append(f"{name}: {'—' if value is None else self._format(value)}")
        return lines

    def _format(self, value: float) -> str:
        return f"{value:.2%}" if self.percent else f"{value:.4g}"

    # Painting -------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignLeft), self.title)
        area = self._area()
        if not any(points for _, points in self.series):
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(area, int(Qt.AlignmentFlag.AlignCenter), "Choose runs with per-epoch curves")
            return
        self._axes(painter, area)
        for index, (name, points) in enumerate(self.series):
            self._line(painter, area, QColor(SERIES[index]), points, name if len(self.series) <= 4 else "")
        self._legend(painter)
        self._hover(painter, area)

    def _axes(self, painter: QPainter, area: QRectF) -> None:
        epochs, low, high = self._ranges()
        painter.setFont(_font(MONO_FAMILY, 10))
        for fraction in (0.0, 0.5, 1.0):
            y = area.bottom() - fraction * area.height()
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(COLORS["text_faint"]))
            value = low + fraction * (high - low)
            painter.drawText(QRectF(0, y - 7, self.LEFT - 6, 14),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             f"{value:.0%}" if self.percent else f"{value:.3g}")
        below = area.bottom() + 3
        painter.drawText(QRectF(area.left(), below, 80, 14), int(Qt.AlignmentFlag.AlignLeft), "epoch 1")
        right = int(Qt.AlignmentFlag.AlignRight)
        painter.drawText(QRectF(area.right() - 80, below, 80, 14), right, str(epochs))

    def _line(self, painter: QPainter, area: QRectF, colour: QColor, points: Sequence[tuple[int, float]],
              label: str) -> None:
        if not points:
            return
        xy = [self._xy(area, *p) for p in points]
        path = QPainterPath(xy[0])
        for point in xy[1:]:
            path.lineTo(point)
        pen = QPen(colour, 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)
        if len(xy) <= 30:
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))
            painter.setBrush(colour)
            for point in xy:
                painter.drawEllipse(point, 4, 4)
        if label:  # direct label at the line's end, in a text colour
            painter.setFont(_font(FONT_FAMILY, 11))
            painter.setPen(QColor(COLORS["text_dim"]))
            metrics = painter.fontMetrics()
            text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, self.RIGHT - 12)
            painter.drawText(QPointF(xy[-1].x() + 8, xy[-1].y() + 4), text)

    def _legend(self, painter: QPainter) -> None:
        painter.setFont(_font(FONT_FAMILY, 11))
        metrics = painter.fontMetrics()
        x, y = float(self.LEFT), self.height() - 14.0
        for index, (name, _points) in enumerate(self.series):
            width = metrics.horizontalAdvance(name) + 26
            if x + width > self.width() and x > self.LEFT:
                break  # the legend never wraps over the plot; direct labels or hover name the rest
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(SERIES[index]))
            painter.drawRoundedRect(QRectF(x, y - 8, 12, 4), 2, 2)
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(QPointF(x + 16, y - 2), name)
            x += width

    def _hover(self, painter: QPainter, area: QRectF) -> None:
        epoch = self.nearest_epoch()
        if epoch is None:
            return
        x = self._xy(area, epoch, 0).x()
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        lines = self.hover_lines()
        painter.setFont(_font(MONO_FAMILY, 11))
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 30
        box = QRectF(min(x + 10, area.right() - width), area.top() + 4, width, 18 * len(lines) + 6)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRoundedRect(box, 3, 3)
        for row, line in enumerate(lines):
            top = box.top() + 4 + 18 * row
            if row:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(SERIES[row - 1]))
                painter.drawRoundedRect(QRectF(box.left() + 8, top + 7, 10, 4), 2, 2)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(QRectF(box.left() + 24, top, width - 26, 18),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), line)
        for index, (_name, points) in enumerate(self.series):
            value = dict(points).get(epoch)
            if value is not None:
                painter.setPen(QPen(QColor(COLORS["surface"]), 2))
                painter.setBrush(QColor(SERIES[index]))
                painter.drawEllipse(self._xy(area, epoch, value), 5, 5)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover_x = event.position().x()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover_x = None
        self.update()
