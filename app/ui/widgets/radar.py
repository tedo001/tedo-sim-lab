"""A radar (spider) chart: several runs' scores on the same measures (accuracy, precision, recall,
F1, ROC AUC; mAP, AP50, mean IoU for detection), one spoke per measure, each run a closed shape in
its own series colour. Every measure shares one axis, 0 at the centre to 1 at the rim; ``zoomed``
runs each spoke from the lowest run's score (the inner ring) to the highest (the rim) instead, so
small differences show (the values are then read by hovering, not from the rings). At most four runs
(more overlapping shapes cannot be read); each is named in the legend, and hovering a corner reads
the value."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY, SERIES

__all__ = ["MAX_SHAPES", "RadarChart"]

MAX_SHAPES = 4
_RINGS = (0.25, 0.5, 0.75, 1.0)
_LEGEND = 22


def _font(family: str, size: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(size)
    return font


class RadarChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.axes: list[str] = []
        #: (name, one value 0..1 per axis, or None when missing)
        self.shapes: list[tuple[str, list[float | None]]] = []
        self.raw: list[list[float | None]] = []
        self.zoomed = False
        self._mouse: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(420, 360)

    def set_data(self, axes: Sequence[str], shapes: Sequence[tuple[str, Sequence[float | None]]], *,
                 zoomed: bool = False) -> None:
        """``shapes``: (run name, value per axis); values are scores where 1 is best."""
        if len(shapes) > MAX_SHAPES:
            raise ValueError(f"at most {MAX_SHAPES} runs: more shapes on one radar cannot be read")
        self.axes = list(axes)
        self.raw = [[None if v is None else float(v) for v in values] for _name, values in shapes]
        self.zoomed = zoomed
        known = [[row[i] for row in self.raw if row[i] is not None] for i in range(len(self.axes))]
        self.shapes = []
        for (name, _values), row in zip(shapes, self.raw, strict=True):
            scaled = [None if v is None else self._zoom(v, known[i]) if zoomed else v
                      for i, v in enumerate(row)]
            self.shapes.append((name, [None if v is None else min(max(v, 0.0), 1.0) for v in scaled]))
        self.update()

    @staticmethod
    def _zoom(value: float, values: list[float]) -> float:
        """The inner ring (0.25) for the lowest score on this spoke, the rim for the highest."""
        low, high = min(values), max(values)
        return 1.0 if high - low < 1e-12 else _RINGS[0] + (1 - _RINGS[0]) * (value - low) / (high - low)

    # Geometry ------------------------------------------------------------------------------
    def _centre_radius(self) -> tuple[QPointF, float]:
        height = self.height() - _LEGEND - 8
        radius = max(min(self.width() / 2 - 120, height / 2 - 30), 20)
        return QPointF(self.width() / 2, 8 + height / 2), radius

    def _point(self, axis: int, value: float) -> QPointF:
        centre, radius = self._centre_radius()
        angle = -math.pi / 2 + 2 * math.pi * axis / len(self.axes)
        return QPointF(centre.x() + math.cos(angle) * radius * value,
                       centre.y() + math.sin(angle) * radius * value)

    def hovered(self) -> tuple[int, int] | None:
        """(shape, axis) of the corner nearest the mouse, within 10 px."""
        if self._mouse is None or len(self.axes) < 3:
            return None
        best, found = 10.0, None
        for s, (_name, values) in enumerate(self.shapes):
            for a, value in enumerate(values):
                if value is None:
                    continue
                point = self._point(a, value)
                distance = math.hypot(point.x() - self._mouse.x(), point.y() - self._mouse.y())
                if distance < best:
                    best, found = distance, (s, a)
        return found

    def hover_text(self) -> str | None:
        found = self.hovered()
        if found is None:
            return None
        s, a = found
        value = self.raw[s][a]
        return f"{self.shapes[s][0]} · {self.axes[a]} {value:.4f}"

    def _label_box(self, axis: int, radius: float) -> tuple[QRectF, Qt.AlignmentFlag]:
        """Where a spoke's name goes: outside the rim, aligned away from the centre."""
        centre, _radius = self._centre_radius()
        outer = self._point(axis, 1.0)
        ux, uy = (outer.x() - centre.x()) / radius, (outer.y() - centre.y()) / radius
        x, y = outer.x() + ux * 10, outer.y() + uy * 10
        width, height = 130.0, 18.0
        if ux > 0.3:
            left, horizontal = x, Qt.AlignmentFlag.AlignLeft
        elif ux < -0.3:
            left, horizontal = x - width, Qt.AlignmentFlag.AlignRight
        else:
            left, horizontal = x - width / 2, Qt.AlignmentFlag.AlignHCenter
        top = y - height if uy < -0.3 else y if uy > 0.3 else y - height / 2
        return QRectF(left, top, width, height), horizontal | Qt.AlignmentFlag.AlignVCenter

    # Painting ------------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if len(self.axes) < 3 or not self.shapes:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.setFont(_font(FONT_FAMILY, 12))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter),
                             "Tick runs that share at least three scores")
            return
        centre, radius = self._centre_radius()
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring in _RINGS:  # recessive rings
            painter.drawPolygon(QPolygonF([self._point(a, ring) for a in range(len(self.axes))]))
        painter.setFont(_font(FONT_FAMILY, 12))
        for a, name in enumerate(self.axes):  # spokes and their names
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawLine(centre, self._point(a, 1.0))
            painter.setPen(QColor(COLORS["text_dim"]))
            box, align = self._label_box(a, radius)
            painter.drawText(box, int(align), name)
        hovered = self.hovered()
        for s, (_name, values) in enumerate(self.shapes):
            colour = QColor(SERIES[s])
            points = [self._point(a, v) for a, v in enumerate(values) if v is not None]
            fill = QColor(colour)
            fill.setAlphaF(0.22 if hovered is None or hovered[0] == s else 0.08)
            painter.setBrush(fill)
            painter.setPen(QPen(colour, 2))
            if len(points) == len(values):
                painter.drawPolygon(QPolygonF(points))
            else:  # a missing score leaves a gap: draw only the edges between known corners
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for a, value in enumerate(values):
                    after = values[(a + 1) % len(values)]
                    if value is not None and after is not None:
                        painter.drawLine(self._point(a, value), self._point((a + 1) % len(values), after))
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))
            painter.setBrush(colour)
            for point in points:
                painter.drawEllipse(point, 4, 4)
        self._ring_values(painter)
        if hovered is not None:
            s, a = hovered
            point = self._point(a, self.shapes[s][1][a])
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))
            painter.setBrush(QColor(SERIES[s]))
            painter.drawEllipse(point, 6, 6)
            text = self.hover_text() or ""
            painter.setFont(_font(MONO_FAMILY, 11))
            width = painter.fontMetrics().horizontalAdvance(text) + 14
            box = QRectF(min(max(point.x() + 10, 4), self.width() - width - 4), point.y() - 26, width, 20)
            painter.setPen(QPen(QColor(COLORS["border"]), 1))
            painter.setBrush(QColor(COLORS["raised"]))
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)
        self._legend(painter)

    def _ring_values(self, painter: QPainter) -> None:
        """Ring values beside the top spoke, over the shapes on small boxes so they stay readable."""
        if self.zoomed:
            return
        painter.setFont(_font(MONO_FAMILY, 10))
        width = painter.fontMetrics().horizontalAdvance("0.00") + 6
        for ring in _RINGS:
            point = self._point(0, ring)
            box = QRectF(point.x() + 8, point.y() - 7, width, 14)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["surface"]))
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), f"{ring:.2f}")

    def _legend(self, painter: QPainter) -> None:
        painter.setFont(_font(FONT_FAMILY, 12))
        x, y = 8.0, self.height() - _LEGEND + 4
        for s, (name, _values) in enumerate(self.shapes):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(SERIES[s]))
            painter.drawRoundedRect(QRectF(x, y + 3, 10, 10), 2, 2)
            painter.setPen(QColor(COLORS["text"]))
            label = painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 150)
            painter.drawText(QRectF(x + 14, y, 160, 16), int(Qt.AlignmentFlag.AlignVCenter), label)
            x += 14 + painter.fontMetrics().horizontalAdvance(label) + 16

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse = event.position()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802, ANN001 (Qt override)
        self._mouse = None
        self.update()
