"""Where the explainer's input comes from: built-in samples, a picture from disk, or a
digit drawn with the mouse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QWidget

from labs.computer_vision.explainer import Sample, prepare_image

from ...theme.tokens import COLORS
from .colours import grey_image, rgb_image
from .maps import MapView

__all__ = ["DrawPad", "SampleStrip", "load_image"]

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"


def load_image(path: str | Path, size: int, channels: int, *, invert: bool = False) -> np.ndarray:
    """A picture from disk, centre-cropped to a square and scaled to ``size`` × ``size`` →
    (C, H, W) in 0..1, as CNN Explainer does with an uploaded image."""
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"Could not read {Path(path).name} as an image")
    side = min(image.width(), image.height())
    image = image.copy((image.width() - side) // 2, (image.height() - side) // 2, side, side)
    image = image.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    stride = image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * size).reshape(size, stride)
    rgb = raw[:, :size * 3].reshape(size, size, 3).copy()
    return prepare_image(rgb, channels, invert=invert)


class SampleStrip(QWidget):
    """Clickable thumbnails of the built-in samples."""

    chosen = Signal(int)

    PER_ROW = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.thumbs: list[MapView] = []

    def set_samples(self, samples: list[Sample]) -> None:
        for thumb in self.thumbs:
            thumb.deleteLater()
        self.thumbs = []
        for index, sample in enumerate(samples):
            thumb = MapView(40, clickable=True)
            thumb.set_image(rgb_image(sample.image))
            thumb.setToolTip(f"{sample.title}\n{sample.credit}")
            thumb.clicked.connect(lambda index=index: self.chosen.emit(index))
            self.grid.addWidget(thumb, index // self.PER_ROW, index % self.PER_ROW)
            self.thumbs.append(thumb)
        self.grid.setColumnStretch(self.PER_ROW, 1)

    def set_current(self, index: int | None) -> None:
        for i, thumb in enumerate(self.thumbs):
            thumb.set_selected(i == index)


class DrawPad(QWidget):
    """A ``size`` × ``size`` canvas: drag to draw white strokes on black, like an MNIST digit.
    Emits the drawing (H × W, 0..1) when a stroke ends."""

    drawn = Signal(object)

    def __init__(self, size: int = 28, side: int = 168, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid_size = size
        self.values = np.zeros((size, size))
        self._last: QPointF | None = None
        self.setFixedSize(side, side)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Draw with the mouse; the network updates when you let go")

    def sizeHint(self) -> QSize:
        return self.size()

    def clear(self) -> None:
        self.values = np.zeros((self.grid_size, self.grid_size))
        self.update()
        self.drawn.emit(self.values.copy())

    def _stamp(self, point: QPointF) -> None:
        """A soft round brush about two cells wide."""
        scale = self.grid_size / self.width()
        cy, cx = point.y() * scale, point.x() * scale
        ys, xs = np.mgrid[0:self.grid_size, 0:self.grid_size] + 0.5
        distance = np.hypot(ys - cy, xs - cx)
        self.values = np.maximum(self.values, np.clip(1.6 - distance, 0, 1))

    def _stroke(self, point: QPointF) -> None:
        start = self._last or point
        steps = max(int((point - start).manhattanLength() / 3), 1)
        for i in range(1, steps + 1):
            self._stamp(start + (point - start) * (i / steps))
        self._last = point
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = None
            self._stroke(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._stroke(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = None
            self.drawn.emit(self.values.copy())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), grey_image(self.values))
        painter.setPen(QColor(COLORS["border"]))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()

