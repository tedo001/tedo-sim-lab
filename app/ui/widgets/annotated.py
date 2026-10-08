"""An image with its annotations drawn on top: boxes, polygons and masks in the one series
colour, each named by a text tag (identity in words, not colours). Hovering an annotation
highlights it and says what it is. Without the image file, the boxes are drawn on a blank
canvas of the image's size."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, MONO_FAMILY

__all__ = ["AnnotatedImage"]


class AnnotatedImage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image: QImage | None = None
        self.size_px = (0, 0)
        self.annotations: list[Any] = []
        self.masks: dict[int, QImage] = {}
        self.hovered: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(480, 360)

    def set_content(self, image: QImage | None, width: int, height: int, annotations: Sequence[Any],
                    masks: dict[int, np.ndarray] | None = None) -> None:
        self.image = image if image is not None and not image.isNull() else None
        self.size_px = (width or (image.width() if image else 0), height or (image.height() if image else 0))
        self.annotations = list(annotations)
        colour = QColor(COLORS["series_1"])
        self.masks = {}
        for index, mask in (masks or {}).items():
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[mask > 0] = (colour.red(), colour.green(), colour.blue(), 90)
            self.masks[index] = QImage(rgba.data, mask.shape[1], mask.shape[0], 4 * mask.shape[1],
                                       QImage.Format.Format_RGBA8888).copy()
        self.hovered = None
        self.update()

    # Geometry -------------------------------------------------------------------
    def _frame(self) -> tuple[QRectF, float]:
        width, height = self.size_px
        if not width or not height:
            return QRectF(), 1.0
        scale = min(self.width() / width, self.height() / height)
        frame = QRectF(0, 0, width * scale, height * scale)
        frame.moveCenter(QRectF(self.rect()).center())
        return frame, scale

    def _box(self, annotation: Any) -> QRectF:
        frame, scale = self._frame()
        x, y, w, h = annotation.bbox
        return QRectF(frame.left() + x * scale, frame.top() + y * scale, w * scale, h * scale)

    def hover_text(self) -> str | None:
        if self.hovered is None:
            return None
        item = self.annotations[self.hovered]
        x, y, w, h = item.bbox
        crowd = " · crowd" if item.crowd else ""
        return f"{item.category} · box {w:.0f}×{h:.0f} at ({x:.0f}, {y:.0f}) · area {item.area:,.0f}{crowd}"

    # Painting -------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame, scale = self._frame()
        if frame.isEmpty():
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Choose an image")
            return
        if self.image is not None:
            painter.drawImage(frame, self.image)
        else:
            painter.fillRect(frame, QColor(COLORS["raised"]))
        for index, mask in self.masks.items():
            if self.hovered in (None, index):
                painter.drawImage(frame, mask)
        series = QColor(COLORS["series_1"])
        for index, item in enumerate(self.annotations):
            emphasis = self.hovered == index
            if self.hovered is not None and not emphasis:
                continue  # hovering one annotation hides the others
            fill = QColor(series)
            fill.setAlpha(70 if emphasis else 40)
            painter.setBrush(fill)
            painter.setPen(QPen(series, 2))
            for polygon in item.polygons:
                points = [QPointF(frame.left() + polygon[i] * scale, frame.top() + polygon[i + 1] * scale)
                          for i in range(0, len(polygon) - 1, 2)]
                painter.drawPolygon(QPolygonF(points))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(COLORS["text"]) if emphasis else series, 2))
            painter.drawRect(self._box(item))
            self._tag(painter, item.category, self._box(item).topLeft())
        text = self.hover_text()
        if text:
            self._tag(painter, text, QPointF(frame.left() + 4, frame.bottom() - 4), bottom=True)

    def _tag(self, painter: QPainter, text: str, at: QPointF, *, bottom: bool = False) -> None:
        font = QFont(MONO_FAMILY)
        font.setPixelSize(11)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 8
        box = QRectF(at.x(), at.y() - (18 if bottom else 0), width, 16)
        box.moveTop(max(box.top(), 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRect(box)
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = event.position()
        inside = [i for i, item in enumerate(self.annotations) if self._box(item).contains(point)]
        # The smallest box under the cursor wins, so nested objects stay reachable.
        found = min(inside, key=lambda i: self.annotations[i].bbox[2] * self.annotations[i].bbox[3]) \
            if inside else None
        if found != self.hovered:
            self.hovered = found
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.hovered = None
        self.update()
