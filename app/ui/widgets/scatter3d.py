"""A point cloud in 3-D (see :mod:`.view3d` for the camera): the data's box with a recessive grid on
its three back walls, ticks and axis names on the near edges, the points far to near, a legend
and the point under the pointer.

Groups follow the lab's scatter rule: up to three get their own validated colour (and their own
marker, so colour is never the only cue); with more, one chosen group is coloured and the rest
fold into a neutral "other". A number to colour by (no groups) uses a one-hue sequential scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..theme.ramps import colour_ramp
from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY
from .view3d import OrbitView, font

__all__ = ["GROUPS", "MARKERS", "ScatterView", "nice_ticks"]

GROUPS = (COLORS["series_1"], COLORS["series_2"], COLORS["series_3"])
MARKERS = ("circle", "square", "triangle")
MAX_POINTS = 6000


def nice_ticks(low: float, high: float, count: int = 5) -> np.ndarray:
    """Round tick values covering [low, high]."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        high = low + 1.0
    raw = (high - low) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    return np.arange(math.floor(low / step), math.ceil(high / step) + 1) * step


def _fmt(value: float) -> str:
    return f"{value:.4g}" if value == 0 or 1e-3 <= abs(value) < 1e5 else f"{value:.2e}"


class ScatterView(OrbitView):
    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.axes = ("x", "y", "z")
        self.names: list[str] = []
        self.highlight = 0
        self.value_label = ""
        self._points = np.zeros((0, 3))
        self._groups: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._ticks: list[np.ndarray] = []
        self._scene = np.zeros((0, 3))
        self._cache: QImage | None = None
        self._screen: np.ndarray | None = None
        self._hover: int | None = None

    # Data -------------------------------------------------------------------------------------------
    def set_points(self, points: np.ndarray, *, groups: Sequence[int] | None = None,
                   names: Sequence[str] = (), values: Sequence[float] | None = None,
                   axes: tuple[str, str, str] = ("x", "y", "z"), value_label: str = "",
                   highlight: int = 0) -> None:
        """``points`` (N × 3, data units); ``groups`` (a group index per point, named by ``names``)
        or ``values`` (a number per point) colour them. At most :data:`MAX_POINTS` are drawn."""
        points = np.asarray(points, float).reshape(-1, 3)
        keep = np.all(np.isfinite(points), axis=1)
        if keep.sum() > MAX_POINTS:
            chosen = np.flatnonzero(keep)
            keep[:] = False
            keep[np.random.default_rng(0).choice(chosen, MAX_POINTS, replace=False)] = True
        self._points = points[keep]
        self._groups = None if groups is None else np.asarray(groups, int)[keep]
        self._values = None if values is None or groups is not None else np.asarray(values, float)[keep]
        self.names, self.axes, self.value_label, self.highlight = list(names), axes, value_label, highlight
        self._ticks = [nice_ticks(float(self._points[:, k].min()), float(self._points[:, k].max()))
                       if len(self._points) else np.array([0.0, 1.0]) for k in range(3)]
        spans = [(t[0], t[-1]) for t in self._ticks]
        self._scene = (np.stack([-1 + 2 * (self._points[:, k] - lo) / ((hi - lo) or 1.0)
                                 for k, (lo, hi) in enumerate(spans)], axis=1)
                       if len(self._points) else np.zeros((0, 3)))
        self._hover = None
        self.refit()

    def set_highlight(self, index: int) -> None:
        self.highlight = max(0, index)
        self.changed()

    def changed(self) -> None:
        self._cache = None
        self.update()

    def corners(self) -> np.ndarray:
        reach = 1.3  # room for the tick labels and axis names around the box
        return np.array([(a, b, c) for a in (-reach, reach) for b in (-reach, reach) for c in (-1.05, 1.05)])

    def folded(self) -> bool:
        return self._groups is not None and len(self.names) > len(GROUPS)

    def colour_of(self, group: int) -> QColor:
        if not self.folded():
            return QColor(GROUPS[group % len(GROUPS)])
        return QColor(COLORS["series_1"] if group == self.highlight else COLORS["series_other"])

    def _value_colours(self) -> list[QColor]:
        ramp = colour_ramp(COLORS["surface_low"], COLORS["surface_high"])
        low, high = float(self._values.min()), float(self._values.max())
        index = np.clip(((self._values - low) / ((high - low) or 1.0) * 255).round().astype(int), 0, 255)
        return [QColor.fromRgbF(*ramp[i]) for i in index]

    # Painting -----------------------------------------------------------------------------------------------
    def _back(self) -> tuple[float, float]:
        """The u and v sides of the box farther from the viewer."""
        eye = self.eye()
        return (-1.0 if eye[0] > 0 else 1.0), (-1.0 if eye[1] > 0 else 1.0)

    def _line(self, painter: QPainter, a: Sequence[float], b: Sequence[float]) -> None:
        ends, _ = self.project(*np.array([a, b], float).T)
        painter.drawLine(QPointF(*ends[0]), QPointF(*ends[1]))

    def _paint_box(self, painter: QPainter) -> None:
        back_u, back_v = self._back()
        level = [-1 + 2 * (t - t[0]) / ((t[-1] - t[0]) or 1.0) for t in self._ticks]
        painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
        for u in level[0]:  # the floor and the two back walls, ruled at the ticks
            self._line(painter, (u, -1, -1), (u, 1, -1))
            self._line(painter, (u, back_v, -1), (u, back_v, 1))
        for v in level[1]:
            self._line(painter, (-1, v, -1), (1, v, -1))
            self._line(painter, (back_u, v, -1), (back_u, v, 1))
        for w in level[2]:
            self._line(painter, (-1, back_v, w), (1, back_v, w))
            self._line(painter, (back_u, -1, w), (back_u, 1, w))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        for a, b in (((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)), ((-1, -1, -1), (-1, 1, -1)),
                     ((1, -1, -1), (1, 1, -1)), ((back_u, back_v, -1), (back_u, back_v, 1)),
                     ((back_u, -1, 1), (back_u, 1, 1)), ((-1, back_v, 1), (1, back_v, 1)),
                     ((-back_u, back_v, -1), (-back_u, back_v, 1)),
                     ((back_u, -back_v, -1), (back_u, -back_v, 1))):
            self._line(painter, a, b)
        front_u, front_v = -back_u, -back_v
        # The upright axis is labelled beside its outer edge, on whichever side of the box it is.
        base, _ = self.project(np.array([back_u, 0.0]), np.array([front_v, 0.0]), np.array([0.0, 0.0]))
        side = -1.0 if base[0][0] < base[1][0] else 1.0
        painter.setFont(font(MONO_FAMILY, 10))
        painter.setPen(QColor(COLORS["text_faint"]))
        for axis, place in ((0, lambda s: (s, front_v * 1.12, -1)), (1, lambda s: (front_u * 1.12, s, -1)),
                            (2, lambda s: (back_u, front_v, s))):
            for value, s in zip(self._ticks[axis], level[axis], strict=True):
                spot = self.project(*np.array([place(s)], float).T)[0][0]
                if axis == 2:
                    spot = spot + np.array([side * 30, 0.0])
                painter.drawText(QRectF(spot[0] - 30, spot[1] - 8, 60, 16), int(Qt.AlignmentFlag.AlignCenter),
                                 _fmt(float(value)))
        painter.setFont(font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        for text, point, shift in ((self.axes[0], (0, front_v * 1.32, -1), 0.0),
                                   (self.axes[1], (front_u * 1.32, 0, -1), 0.0),
                                   (self.axes[2], (back_u, front_v, 0), side * 84)):
            spot = self.project(*np.array([point], float).T)[0][0] + np.array([shift, 0.0])
            painter.drawText(QRectF(spot[0] - 70, spot[1] - 9, 140, 18), int(Qt.AlignmentFlag.AlignCenter),
                             text)

    def _marker(self, painter: QPainter, centre: QPointF, kind: str, r: float) -> None:
        if kind == "square":
            painter.drawRect(QRectF(centre.x() - r * 0.9, centre.y() - r * 0.9, r * 1.8, r * 1.8))
        elif kind == "triangle":
            painter.drawPolygon(QPolygonF([QPointF(centre.x(), centre.y() - r * 1.15),
                                           QPointF(centre.x() + r * 1.05, centre.y() + r * 0.8),
                                           QPointF(centre.x() - r * 1.05, centre.y() + r * 0.8)]))
        else:
            painter.drawEllipse(centre, r, r)

    def _paint_points(self, painter: QPainter) -> None:
        if not len(self._scene):
            return
        screen, depth = self.project(self._scene[:, 0], self._scene[:, 1], self._scene[:, 2])
        self._screen = screen
        order = np.argsort(-depth)
        if self.folded():  # the coloured group last, on top of the folded ones
            order = order[np.argsort(self._groups[order] == self.highlight, kind="stable")]
        values = self._value_colours() if self._values is not None else None
        radius = 3.2 if len(self._scene) < 1500 else 2.6
        painter.setPen(QPen(QColor(COLORS["surface"]), 0.8))
        for index in order.tolist():
            if values is not None:
                colour, kind = values[index], "circle"
            elif self._groups is not None:
                group = int(self._groups[index])
                colour = self.colour_of(group)
                folded = self.folded() and group != self.highlight
                kind = "circle" if folded else MARKERS[group % len(MARKERS)]
            else:
                colour, kind = QColor(COLORS["series_1"]), "circle"
            colour.setAlphaF(0.9)
            painter.setBrush(colour)
            self._marker(painter, QPointF(*screen[index]), kind, radius)

    def _paint_legend(self, painter: QPainter) -> None:
        painter.setFont(font(FONT_FAMILY, 11))
        right, top = self.width() - 8, 22
        if self._values is not None:
            ramp = colour_ramp(COLORS["surface_low"], COLORS["surface_high"], steps=8)
            for index, rgb in enumerate(ramp):
                painter.fillRect(QRectF(right - (8 - index) * 14, top + 16, 13, 8), QColor.fromRgbF(*rgb))
            painter.setPen(QColor(COLORS["text_dim"]))
            low, high = float(self._values.min()), float(self._values.max())
            painter.drawText(QRectF(right - 260, top, 260, 14), int(Qt.AlignmentFlag.AlignRight),
                             f"{self.value_label}: {_fmt(low)} … {_fmt(high)}")
            return
        if self._groups is None or not self.names:
            return
        rows = ([(self.names[self.highlight], self.highlight), ("other groups", -1)] if self.folded()
                else [(name, index) for index, name in enumerate(self.names)])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for row, (name, group) in enumerate(rows):
            y = top + row * 17
            colour = QColor(COLORS["series_other"]) if group < 0 else self.colour_of(group)
            kind = "circle" if group < 0 or self.folded() else MARKERS[group % len(MARKERS)]
            painter.setPen(QPen(QColor(COLORS["surface"]), 0.8))
            painter.setBrush(colour)
            self._marker(painter, QPointF(right - 6, y + 7), kind, 4)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(QRectF(right - 220, y, 204, 15), int(Qt.AlignmentFlag.AlignRight), name)

    def _render(self) -> QImage:
        ratio = self.devicePixelRatioF()
        image = QImage(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)),
                       QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_box(painter)
        self._paint_points(painter)
        self._paint_legend(painter)
        painter.end()
        return image

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(4, 0, self.width() - 8, 18), int(Qt.AlignmentFlag.AlignLeft), self.title)
        if not len(self._points):
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Nothing to show yet")
            return
        if self._cache is None or self._cache.deviceIndependentSize().toSize() != self.size():
            self._cache = self._render()
        painter.drawImage(0, 0, self._cache)
        if self._hover is not None and self._screen is not None:
            index = self._hover
            spot = QPointF(*self._screen[index])
            painter.setPen(QPen(QColor(COLORS["text"]), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(spot, 6, 6)
            parts = [f"{name} {_fmt(float(value))}"
                     for name, value in zip(self.axes, self._points[index], strict=True)]
            if self._groups is not None and 0 <= int(self._groups[index]) < len(self.names):
                parts.append(self.names[int(self._groups[index])])
            elif self._values is not None:
                parts.append(f"{self.value_label} {_fmt(float(self._values[index]))}")
            painter.setFont(font(MONO_FAMILY, 11))
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(QRectF(4, self.height() - 18, self.width() - 8, 16),
                             int(Qt.AlignmentFlag.AlignLeft), " · ".join(parts))

    def hover_at(self, position: QPointF | None) -> None:
        found = None
        if position is not None and self._screen is not None and len(self._screen):
            distance = np.hypot(self._screen[:, 0] - position.x(), self._screen[:, 1] - position.y())
            nearest = int(np.argmin(distance))
            found = nearest if distance[nearest] < 10 else None
        if found != self._hover:
            self._hover = found
            self.update()
