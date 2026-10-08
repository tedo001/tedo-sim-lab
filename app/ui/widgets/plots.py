"""Charts for fitted-model results, drawn with QPainter by the lab's chart rules: one axis,
recessive grid, values in text colours, hover on the nearest mark.

- :class:`CurveChart`: one curve on 0..1 axes (ROC, precision-recall), optional chance line.
- :class:`ScatterChart`: a point cloud (predicted vs actual, clusters, projections). Up to three
  groups get their own colour (the validated all-pairs slots); with more, one chosen group is
  coloured and the rest fold into a neutral "other".
- :class:`BarList`: labelled horizontal bars (feature importance), largest first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["BarList", "CurveChart", "GROUP_COLOURS", "ScatterChart"]

GROUP_COLOURS = (COLORS["series_1"], COLORS["series_2"], COLORS["series_3"])


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


def _nice(value: float) -> str:
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e5 else f"{value:.1e}"


class _Plot(QWidget):
    LEFT, RIGHT, TOP, BOTTOM = 46, 12, 26, 34

    def __init__(self, title: str, x_label: str, y_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title, self.x_label, self.y_label = title, x_label, y_label
        self.headline = ""
        self.x_range, self.y_range = (0.0, 1.0), (0.0, 1.0)
        self._mouse: QPointF | None = None
        self._empty = "Nothing to show"
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(320, 260)

    def _area(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(self.width() - self.LEFT - self.RIGHT, 1),
                      max(self.height() - self.TOP - self.BOTTOM, 1))

    def _xy(self, area: QRectF, x: float, y: float) -> QPointF:
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        return QPointF(area.left() + (x - x0) / ((x1 - x0) or 1) * area.width(),
                       area.bottom() - (y - y0) / ((y1 - y0) or 1) * area.height())

    def _has_data(self) -> bool:
        return False

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignLeft), self.title)
        painter.setFont(_font(MONO_FAMILY, 12))
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignRight), self.headline)
        area = self._area()
        if not self._has_data():
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(area, int(Qt.AlignmentFlag.AlignCenter), self._empty)
            return
        self._paint_axes(painter, area)
        self._paint_marks(painter, area)
        self._paint_hover(painter, area)

    def _paint_axes(self, painter: QPainter, area: QRectF) -> None:
        painter.setFont(_font(MONO_FAMILY, 10))
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        for fraction in (0.0, 0.5, 1.0):
            y = area.bottom() - fraction * area.height()
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(QRectF(0, y - 7, self.LEFT - 6, 14),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             _nice(y0 + fraction * (y1 - y0)))
        bottom = area.bottom() + 6
        painter.drawText(QRectF(area.left(), bottom, 80, 14), int(Qt.AlignmentFlag.AlignLeft), _nice(x0))
        painter.drawText(QRectF(area.right() - 80, bottom, 80, 14), int(Qt.AlignmentFlag.AlignRight),
                         _nice(x1))
        painter.setFont(_font(FONT_FAMILY, 11))
        painter.drawText(QRectF(area.left(), bottom + 2, area.width(), 28),
                         int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
                         f"{self.x_label}  ·  vertical: {self.y_label}")

    def _paint_marks(self, painter: QPainter, area: QRectF) -> None: ...

    def _nearest(self, area: QRectF) -> tuple[QPointF, str, QColor] | None:
        return None

    def _paint_hover(self, painter: QPainter, area: QRectF) -> None:
        found = self._nearest(area) if self._mouse is not None else None
        if found is None:
            return
        point, text, colour = found
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawLine(QPointF(point.x(), area.top()), QPointF(point.x(), area.bottom()))
        painter.setPen(QPen(QColor(COLORS["surface"]), 2))
        painter.setBrush(colour)
        painter.drawEllipse(point, 5, 5)
        painter.setFont(_font(MONO_FAMILY, 11))
        width = painter.fontMetrics().horizontalAdvance(text) + 14
        box = QRectF(min(max(point.x() + 8, area.left()), area.right() - width), area.top() + 2, width, 20)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def hover_text(self) -> str | None:
        found = self._nearest(self._area()) if self._mouse is not None else None
        return None if found is None else found[1]

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse = event.position()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._mouse = None
        self.update()


class CurveChart(_Plot):
    """One curve on 0..1 axes; ``chance`` draws the diagonal a random ranking would follow."""

    def __init__(self, title: str, x_label: str, y_label: str, *, chance: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(title, x_label, y_label, parent)
        self.chance = chance
        self._points: list[tuple[float, float]] = []

    def set_curve(self, points: Sequence[Sequence[float]], headline: str = "") -> None:
        self._points = [(float(x), float(y)) for x, y in points]
        self.headline = headline
        self.update()

    def _has_data(self) -> bool:
        return bool(self._points)

    def _paint_marks(self, painter: QPainter, area: QRectF) -> None:
        if self.chance:
            pen = QPen(QColor(COLORS["text_faint"]), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(self._xy(area, 0, 0), self._xy(area, 1, 1))
        points = [self._xy(area, x, y) for x, y in self._points]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        pen = QPen(QColor(COLORS["series_1"]), 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.strokePath(path, pen)

    def _nearest(self, area: QRectF) -> tuple[QPointF, str, QColor] | None:
        if not self._points or self._mouse is None:
            return None
        x, y = min(self._points, key=lambda p: abs(self._xy(area, *p).x() - self._mouse.x()))
        text = f"{self.x_label} {x:.3f} · {self.y_label} {y:.3f}"
        return self._xy(area, x, y), text, QColor(COLORS["series_1"])


class ScatterChart(_Plot):
    """``points`` are (x, y) or (x, y, group index); ``groups`` names the groups."""

    def __init__(self, title: str, x_label: str, y_label: str, *, identity: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(title, x_label, y_label, parent)
        self.identity = identity
        self._points: list[tuple[float, float, int]] = []
        self.groups: list[str] = []
        self.highlight = 0

    def set_points(self, points: Sequence[Sequence[float]], groups: Sequence[str] = (), headline: str = "",
                   highlight: int = 0) -> None:
        self._points = [(float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 0) for p in points]
        self.groups, self.headline, self.highlight = list(groups), headline, highlight
        if self._points:
            xs, ys = [p[0] for p in self._points], [p[1] for p in self._points]
            if self.identity:  # same scale on both axes, so the diagonal means "exactly right"
                low, high = min(xs + ys), max(xs + ys)
                self.x_range = self.y_range = (low, high if high > low else low + 1)
            else:
                self.x_range = (min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
                self.y_range = (min(ys), max(ys) if max(ys) > min(ys) else min(ys) + 1)
        self.update()

    def colour_of(self, group: int) -> QColor:
        if len(self.groups) <= len(GROUP_COLOURS):
            return QColor(GROUP_COLOURS[group % len(GROUP_COLOURS)])
        return QColor(COLORS["series_1"] if group == self.highlight else COLORS["series_other"])

    def _has_data(self) -> bool:
        return bool(self._points)

    def _paint_marks(self, painter: QPainter, area: QRectF) -> None:
        if self.identity:
            painter.setPen(QPen(QColor(COLORS["text_faint"]), 1, Qt.PenStyle.DashLine))
            low = self.x_range[0]
            high = self.x_range[1]
            painter.drawLine(self._xy(area, low, low), self._xy(area, high, high))
        folded = len(self.groups) > len(GROUP_COLOURS)
        # Folded groups first, so the highlighted one is drawn on top.
        order = sorted(self._points, key=lambda p: folded and p[2] == self.highlight)
        painter.setPen(QPen(QColor(COLORS["surface"]), 1))
        for x, y, group in order:
            colour = self.colour_of(group)
            colour.setAlphaF(0.85)
            painter.setBrush(colour)
            painter.drawEllipse(self._xy(area, x, y), 3.5, 3.5)

    def _nearest(self, area: QRectF) -> tuple[QPointF, str, QColor] | None:
        if not self._points or self._mouse is None:
            return None
        mouse = self._mouse
        x, y, group = min(self._points, key=lambda p: (self._xy(area, p[0], p[1]) - mouse).manhattanLength())
        point = self._xy(area, x, y)
        if (point - mouse).manhattanLength() > 24:
            return None
        name = f" · {self.groups[group]}" if self.groups and group < len(self.groups) else ""
        return point, f"{_nice(x)}, {_nice(y)}{name}", self.colour_of(group)


class BarList(QWidget):
    """Rows of ``(label, value, spread)``: name, bar from zero, value ± spread in text."""

    ROW = 22

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, float, float]] = []
        self._hover: int | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_rows(self, rows: Sequence[Sequence], limit: int = 15) -> None:
        self._rows = [(str(r[0]), float(r[1]), float(r[2]) if len(r) > 2 else 0.0) for r in rows[:limit]]
        self.setFixedHeight(max(len(self._rows), 1) * self.ROW + 4)
        self.update()

    def rows(self) -> list[tuple[str, float, float]]:
        return list(self._rows)

    def sizeHint(self) -> QSize:
        return QSize(320, max(len(self._rows), 1) * self.ROW + 4)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._rows:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Nothing to show")
            return
        name_width = min(180, max(80, self.width() // 3))
        value_width = 110
        bar_left = name_width + 8
        bar_width = max(self.width() - bar_left - value_width - 8, 10)
        top = max(max(value for _, value, _ in self._rows), 1e-12)
        for index, (name, value, spread) in enumerate(self._rows):
            y = index * self.ROW + 2
            if index == self._hover:
                painter.fillRect(QRectF(0, y, self.width(), self.ROW), QColor(COLORS["raised"]))
            painter.setFont(_font(MONO_FAMILY, 11))
            painter.setPen(QColor(COLORS["text_dim"]))
            metrics = painter.fontMetrics()
            painter.drawText(QRectF(0, y, name_width, self.ROW),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             metrics.elidedText(name, Qt.TextElideMode.ElideRight, name_width))
            length = max(value, 0.0) / top * bar_width
            if length > 0:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(COLORS["series_1"]))
                painter.drawRoundedRect(QRectF(bar_left, y + 6, max(length, 2), self.ROW - 12), 2, 2)
            painter.setPen(QColor(COLORS["text"]))
            text = _nice(value) + (f" ± {_nice(spread)}" if spread else "")
            painter.drawText(QRectF(bar_left + bar_width + 8, y, value_width, self.ROW),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        index = int((event.position().y() - 2) // self.ROW)
        self._hover = index if 0 <= index < len(self._rows) else None
        self.setToolTip(self._rows[self._hover][0] if self._hover is not None else "")
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = None
        self.update()
