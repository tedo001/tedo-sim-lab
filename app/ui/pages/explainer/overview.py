"""The whole network at a glance, as in CNN Explainer's overview: one column per layer,
one small map per channel, faint links that show how layers connect, and the class
scores at the end. Hover a map to see its inputs; click it to take it apart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from labs.computer_vision.explainer import Trace

from ...theme.tokens import COLORS, FONT_FAMILY
from .colours import grey_image, heat_image, mono_font, paint_legend

__all__ = ["INPUT", "NetworkOverview"]

#: Column id of the input layer (other columns use the layer's index).
INPUT = -1
_HEAD, _LEGEND, _PAD = 34, 40, 4


@dataclass(frozen=True)
class _Column:
    layer: int
    kind: str
    title: str
    nodes: int
    shape: tuple[int, ...]


def describe(trace: Trace, layer: int, node: int) -> str:
    """One line about a node, for the hover caption."""
    architecture = trace.net.architecture
    if layer == INPUT:
        channel = ("red", "green", "blue")[node] if architecture.input_shape[0] == 3 else "grey"
        values = trace.input[node]
        return f"input · {channel} channel · {values.shape[0]}×{values.shape[1]} · values 0 … 1"
    spec = architecture.layers[layer]
    if spec.kind == "linear":
        return (f"{spec.name} · {architecture.class_names[node]} · score {trace.logits[node]:.4f} · "
                f"probability {trace.probabilities[node]:.4f}")
    values = trace.outputs[layer][node]
    return (f"{spec.name} · channel {node + 1} of {trace.outputs[layer].shape[0]} · "
            f"{values.shape[0]}×{values.shape[1]} · min {values.min():.3f} · max {values.max():.3f}")


class NetworkOverview(QWidget):
    node_clicked = Signal(int, int)
    hover_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.trace: Trace | None = None
        self.limits: dict[int, float] = {}
        self.columns: list[_Column] = []
        self._images: dict[tuple[int, int], QImage] = {}
        self._rects: dict[tuple[int, int], QRectF] = {}
        self._column_x: list[tuple[float, float]] = []
        self._hover: tuple[int, int] | None = None
        self._selected: tuple[int, int] | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(280)
        self.setFixedHeight(300)

    # Data -------------------------------------------------------------------
    def set_trace(self, trace: Trace, limits: dict[int, float]) -> None:
        self.trace, self.limits = trace, limits
        architecture = trace.net.architecture
        shape = architecture.input_shape
        self.columns = [_Column(INPUT, "input", "input", shape[0], shape)]
        for index, layer in enumerate(architecture.layers):
            if layer.is_map:
                self.columns.append(_Column(index, layer.kind, layer.kind, architecture.shapes[index][0],
                                            architecture.shapes[index]))
        output = len(architecture.layers) - 1
        classes = architecture.num_classes
        self.columns.append(_Column(output, "linear", "output", classes, (classes,)))
        self._images = {(INPUT, c): grey_image(trace.input[c]) for c in range(trace.input.shape[0])}
        for column in self.columns[1:-1]:
            maps = trace.outputs[column.layer]
            for c in range(column.nodes):
                self._images[(column.layer, c)] = heat_image(maps[c], limits[column.layer])
        self._layout()
        self.update()

    def set_selected(self, layer: int, node: int) -> None:
        self._selected = (layer, node)
        self.update()

    def node_rect(self, layer: int, node: int) -> QRectF:
        return self._rects[(layer, node)]

    # Geometry ---------------------------------------------------------------
    def _layout(self) -> None:
        if not self.columns:
            return
        maps = len(self.columns) - 1
        longs = sum(1 for column in self.columns[1:] if column.kind in ("conv", "linear"))
        shorts = maps - longs
        width = self.width() - 2 * _PAD
        out_w = min(max(width * 0.16, 84.0), 150.0)
        side = (width - out_w) / (maps + 0.45 * (shorts + 2.2 * longs))
        side = min(max(side, 12.0), 44.0)
        gap = 0.45 * side
        tallest = max(column.nodes for column in self.columns)
        vgap = 0.24 * side
        body = tallest * side + (tallest + 1) * vgap
        self._rects.clear()
        self._column_x.clear()
        x = float(_PAD)
        for number, column in enumerate(self.columns):
            if number:
                x += gap * (2.2 if column.kind in ("conv", "linear") else 1.0)
            column_w = out_w if column.kind == "linear" else side
            self._column_x.append((x, column_w))
            spacing = (body - column.nodes * side) / (column.nodes + 1)
            for node in range(column.nodes):
                top = _HEAD + spacing * (node + 1) + side * node
                self._rects[(column.layer, node)] = QRectF(x, top, column_w, side)
            x += column_w
        height = int(_HEAD + body + _LEGEND + _PAD)
        if height != self.height():
            self.setFixedHeight(height)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._layout()
        super().resizeEvent(event)

    def _hit(self, point: QPointF) -> tuple[int, int] | None:
        for key, rect in self._rects.items():
            if rect.adjusted(-2, -2, 2, 2).contains(point):
                return key
        return None

    def _sources(self, layer: int, node: int) -> list[tuple[int, int]]:
        """The nodes feeding (layer, node)."""
        number = next(i for i, column in enumerate(self.columns) if column.layer == layer)
        if number == 0:
            return []
        previous = self.columns[number - 1]
        if self.columns[number].kind in ("conv", "linear"):
            return [(previous.layer, i) for i in range(previous.nodes)]
        return [(previous.layer, node)]

    # Painting ---------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self.trace is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_links(painter)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self._paint_heads(painter)
        for key, image in self._images.items():
            painter.drawImage(self._rects[key], image)
        self._paint_output(painter)
        self._paint_outlines(painter)
        self._paint_legends(painter)
        painter.end()

    def _paint_links(self, painter: QPainter) -> None:
        faint = QPen(QColor(COLORS["border"]), 0.7)
        strong = QPen(QColor(COLORS["text_dim"]), 1.2)
        hover_sources = set(self._sources(*self._hover)) if self._hover else set()
        for column in self.columns[1:]:
            for node in range(column.nodes):
                end = self._rects[(column.layer, node)]
                for source in self._sources(column.layer, node):
                    start = self._rects[source]
                    lit = self._hover == (column.layer, node) and source in hover_sources
                    painter.setPen(strong if lit else faint)
                    painter.drawLine(QPointF(start.right(), start.center().y()),
                                     QPointF(end.left(), end.center().y()))

    def _paint_heads(self, painter: QPainter) -> None:
        metrics = QFontMetrics(mono_font(10))
        for column, (x, width) in zip(self.columns, self._column_x, strict=True):
            centre = x + width / 2
            room = max(width + 0.45 * width, 24.0)
            painter.setFont(mono_font(10, QFont.Weight.Medium))
            painter.setPen(QColor(COLORS["text_dim"]))
            title = metrics.elidedText(column.title, Qt.TextElideMode.ElideRight, int(room))
            painter.drawText(QRectF(centre - room / 2, 2, room, 14), int(Qt.AlignmentFlag.AlignCenter), title)
            if len(column.shape) == 3:
                size = f"{column.shape[1]}×{column.shape[2]}"
                if metrics.horizontalAdvance(size) <= room:
                    painter.setFont(mono_font(10))
                    painter.setPen(QColor(COLORS["text_faint"]))
                    painter.drawText(QRectF(centre - room / 2, 16, room, 14),
                                     int(Qt.AlignmentFlag.AlignCenter), size)

    def _paint_output(self, painter: QPainter) -> None:
        column = self.columns[-1]
        probabilities = self.trace.probabilities
        top = int(np.argmax(probabilities))
        label_font = QFont(FONT_FAMILY)
        label_font.setPixelSize(11)
        metrics = QFontMetrics(label_font)
        for node in range(column.nodes):
            rect = self._rects[(column.layer, node)]
            name = self.trace.net.architecture.class_names[node]
            painter.setFont(label_font)
            painter.setPen(QColor(COLORS["text"] if node == top else COLORS["text_faint"]))
            text_rect = QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.55)
            value = f"{probabilities[node]:.2f}" if node == top else ""
            value_w = QFontMetrics(mono_font(10)).horizontalAdvance(value) + 4 if value else 0
            painter.drawText(text_rect.adjusted(0, 0, -value_w, 0),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom),
                             metrics.elidedText(name, Qt.TextElideMode.ElideRight,
                                                int(rect.width() - value_w)))
            if value:
                painter.setFont(mono_font(10))
                painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                                 value)
            bar = QRectF(rect.left(), rect.top() + rect.height() * 0.62, rect.width(),
                         max(rect.height() * 0.22, 3))
            painter.fillRect(bar, QColor(COLORS["border_soft"]))
            filled = QRectF(bar.left(), bar.top(), bar.width() * float(probabilities[node]), bar.height())
            painter.fillRect(filled, QColor(COLORS["series_1"]))

    def _paint_outlines(self, painter: QPainter) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._hover is not None:
            painter.setPen(QPen(QColor(COLORS["text_dim"]), 1))
            for key in [self._hover, *self._sources(*self._hover)]:
                painter.drawRect(self._rects[key].adjusted(-1.5, -1.5, 1.5, 1.5))
        if self._selected in self._rects:
            painter.setPen(QPen(QColor(COLORS["accent_hover"]), 2))
            painter.drawRect(self._rects[self._selected].adjusted(-2, -2, 2, 2))

    def _paint_legends(self, painter: QPainter) -> None:
        top = self.height() - _LEGEND - _PAD + 6
        x, width = self._column_x[0]
        paint_legend(painter, QRectF(x, top, max(width, 40), 30), None, title="input")
        focus = self._hover or self._selected
        layer = focus[0] if focus and focus[0] in self.limits else self.columns[1].layer
        first_x = self._column_x[1][0]
        last_x, last_w = self._column_x[-2]
        span = min(last_x + last_w - first_x, 260.0)
        name = self.trace.net.architecture.layers[layer].name
        paint_legend(painter, QRectF(first_x, top, span, 30), self.limits[layer], title=f"{name} activations")

    # Interaction ------------------------------------------------------------
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        hit = self._hit(event.position())
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
            self.hover_changed.emit(describe(self.trace, *hit) if hit and self.trace else "")
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._hover is not None:
            self._hover = None
            self.hover_changed.emit("")
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        hit = self._hit(event.position())
        if hit is not None and event.button() == Qt.MouseButton.LeftButton:
            self.set_selected(*hit)
            self.node_clicked.emit(*hit)
