"""A single activation map (or input channel, or image) drawn pixel for pixel, with the
kernel window and the current cell outlined, and the cell under the pointer reported."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...theme.tokens import COLORS
from .colours import grey_image, heat_image

__all__ = ["MapView"]


class MapView(QWidget):
    """``values`` (H × W) on the diverging scale (``limit``) or as greys (``limit=None``)."""

    hovered = Signal(int, int)
    clicked = Signal()

    def __init__(self, side: int = 160, *, clickable: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._shape = (1, 1)
        self.values: np.ndarray | None = None
        self.limit: float | None = None
        self._window: tuple[int, int, int, int] | None = None
        self._marker: tuple[int, int] | None = None
        self._selected = False
        self.clickable = clickable
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.set_side(side)
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_side(self, side: int) -> None:
        self.setFixedSize(side, side)

    def sizeHint(self) -> QSize:
        return self.size()

    def set_map(self, values: np.ndarray, limit: float | None) -> None:
        self.values = np.asarray(values)
        self.limit = limit
        self._shape = self.values.shape
        self._image = grey_image(self.values) if limit is None else heat_image(self.values, limit)
        self.update()

    def set_image(self, image: QImage, values: np.ndarray | None = None) -> None:
        """Show a ready-made image (the colour input), optionally with values for hovering."""
        self._image = image
        self._shape = (image.height(), image.width())
        self.values = values
        self.update()

    def set_window(self, window: tuple[int, int, int, int] | None) -> None:
        """(top, left, height, width) in cells; may reach outside the map (padding)."""
        self._window = window
        self.update()

    def set_marker(self, cell: tuple[int, int] | None) -> None:
        self._marker = cell
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def cell_rect(self, top: float, left: float, height: float = 1, width: float = 1) -> QRectF:
        rows, columns = self._shape
        sy, sx = self.height() / rows, self.width() / columns
        return QRectF(left * sx, top * sy, width * sx, height * sy)

    def cell_at(self, point: QPointF) -> tuple[int, int] | None:
        rows, columns = self._shape
        y, x = int(point.y() * rows / max(self.height(), 1)), int(point.x() * columns / max(self.width(), 1))
        return (y, x) if 0 <= y < rows and 0 <= x < columns else None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if self._image is not None:
            painter.drawImage(QRectF(self.rect()), self._image)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self._window is not None:
            painter.setPen(QPen(QColor(COLORS["text"]), 2))
            painter.drawRect(self.cell_rect(*self._window).intersected(bounds))
        if self._marker is not None:
            painter.setPen(QPen(QColor(COLORS["text"]), 2))
            painter.drawRect(self.cell_rect(*self._marker).intersected(bounds))
        if self._selected:
            painter.setPen(QPen(QColor(COLORS["accent_hover"]), 2))
            painter.drawRect(bounds)
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cell = self.cell_at(event.position())
        if cell is not None:
            self.hovered.emit(*cell)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
