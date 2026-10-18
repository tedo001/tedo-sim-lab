"""Views the Simulation Lab draws with: an image (numpy → QImage, fitted, pixels kept sharp when
enlarged, optional boxes on top), a physics line chart (one series, optional fill and a shaded
x-band), and a Bloch sphere for one qubit."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, MONO_FAMILY, SERIES
from .plots import _font, _nice, _Plot

__all__ = ["BlochView", "ImageView", "LineChart", "to_qimage"]


def to_qimage(image: np.ndarray) -> QImage:
    """A float image in [0, 1] (grey or RGB) or uint8 as a QImage (a copy that owns its pixels)."""
    pixels = image if image.dtype == np.uint8 else (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    if pixels.ndim == 2:
        return QImage(pixels.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    pixels = np.ascontiguousarray(pixels[..., :3])
    return QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """An image fitted to the widget; ``boxes`` are (x, y, w, h, label) in image pixels."""

    def __init__(self, caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.caption = caption
        self.image: QImage | None = None
        self.boxes: list[tuple[float, float, float, float, str]] = []
        self.setMinimumSize(160, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: np.ndarray | QImage | None, boxes: Sequence = ()) -> None:
        self.image = to_qimage(image) if isinstance(image, np.ndarray) else image
        self.boxes = list(boxes)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(320, 260)

    def image_rect(self) -> QRectF:
        top = 20 if self.caption else 0
        if self.image is None or self.image.isNull():
            return QRectF(0, top, self.width(), self.height() - top)
        area = QRectF(0, top, self.width(), self.height() - top)
        scale = min(area.width() / self.image.width(), area.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF(area.left() + (area.width() - w) / 2, area.top() + (area.height() - h) / 2, w, h)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        if self.caption:
            painter.setFont(_font(MONO_FAMILY, 11))
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignLeft), self.caption)
        rect = self.image_rect()
        if self.image is None or self.image.isNull():
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "No image")
            return
        enlarged = rect.width() > self.image.width()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not enlarged)
        painter.drawImage(rect, self.image)
        if self.boxes:
            scale = rect.width() / self.image.width()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setFont(_font(MONO_FAMILY, 10))
            for x, y, w, h, text in self.boxes:
                box = QRectF(rect.left() + x * scale, rect.top() + y * scale, w * scale, h * scale)
                painter.setPen(QPen(QColor(COLORS["accent_hover"]), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(box)
                if text:
                    painter.setPen(QColor(COLORS["text"]))
                    painter.drawText(box.adjusted(2, 1, 0, 0), int(Qt.AlignmentFlag.AlignLeft), text)


class LineChart(_Plot):
    """One line, x and y on their own ranges; ``fill`` shades under it, ``band`` shades an x-range
    (where a barrier stands)."""

    def __init__(self, title: str, x_label: str, y_label: str, parent: QWidget | None = None) -> None:
        super().__init__(title, x_label, y_label, parent)
        self.xs = np.zeros(0)
        self.ys = np.zeros(0)
        self.fill = False
        self.band: tuple[float, float] | None = None
        self.setMinimumHeight(170)

    def set_line(self, xs: np.ndarray, ys: np.ndarray, *, x_range: tuple[float, float] | None = None,
                 y_range: tuple[float, float] | None = None, fill: bool = False, headline: str = "",
                 band: tuple[float, float] | None = None) -> None:
        step = max(1, len(xs) // 1200)  # at most ~1200 points per frame
        self.xs, self.ys = np.asarray(xs)[::step], np.asarray(ys)[::step]
        self.x_range = x_range or (float(self.xs.min()), float(self.xs.max())) if len(self.xs) else (0.0, 1.0)
        if y_range is None and len(self.ys):
            low, high = float(self.ys.min()), float(self.ys.max())
            y_range = (min(low, 0.0), high if high > low else low + 1.0)
        self.y_range = y_range or (0.0, 1.0)
        self.fill, self.headline, self.band = fill, headline, band
        self.update()

    def _has_data(self) -> bool:
        return len(self.xs) > 1

    def _paint_marks(self, painter: QPainter, area: QRectF) -> None:
        if self.band is not None:
            left, right = (self._xy(area, x, self.y_range[0]).x() for x in self.band)
            painter.fillRect(QRectF(left, area.top(), max(right - left, 1), area.height()),
                             QColor(COLORS["raised"]))
        y_low, y_high = self.y_range
        clipped = np.clip(self.ys, y_low, y_high)
        points = [self._xy(area, float(x), float(y)) for x, y in zip(self.xs, clipped, strict=True)]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        colour = QColor(SERIES[0])
        if self.fill:
            area_path = QPainterPath(path)
            base = max(y_low, min(0.0, y_high))
            area_path.lineTo(self._xy(area, float(self.xs[-1]), base))
            area_path.lineTo(self._xy(area, float(self.xs[0]), base))
            area_path.closeSubpath()
            tint = QColor(colour)
            tint.setAlpha(46)
            painter.fillPath(area_path, tint)
        pen = QPen(colour, 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)

    def _nearest(self, area: QRectF):
        if not self._has_data() or self._mouse is None:
            return None
        x0, x1 = self.x_range
        x = x0 + (self._mouse.x() - area.left()) / max(area.width(), 1) * (x1 - x0)
        index = int(np.argmin(np.abs(self.xs - x)))
        value = float(np.clip(self.ys[index], *self.y_range))
        x_text, y_text = _nice(float(self.xs[index])), _nice(float(self.ys[index]))
        text = f"{self.x_label} {x_text} · {self.y_label} {y_text}"
        return self._xy(area, float(self.xs[index]), value), text, QColor(SERIES[0])


class BlochView(QWidget):
    """One qubit on the Bloch sphere: |0⟩ at the top, |1⟩ at the bottom; a vector shorter than
    the radius means the qubit is entangled with others (its state is mixed)."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.vector = (0.0, 0.0, 1.0)
        self.setMinimumSize(132, 150)
        self.setToolTip("Bloch sphere: north |0⟩, south |1⟩, east |+⟩; a short vector = entangled")

    def set_vector(self, vector: tuple[float, float, float], title: str | None = None) -> None:
        self.vector = vector
        if title is not None:
            self.title = title
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(150, 170)

    def _project(self, x: float, y: float, z: float, centre: QPointF, radius: float) -> QPointF:
        azimuth, elevation = math.radians(-35), math.radians(18)  # the viewer's position
        px = x * math.cos(azimuth) - y * math.sin(azimuth)
        depth = x * math.sin(azimuth) + y * math.cos(azimuth)
        py = z * math.cos(elevation) - depth * math.sin(elevation)
        return QPointF(centre.x() + px * radius, centre.y() - py * radius)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = max(min(self.width(), self.height() - 36) / 2 - 12, 10)
        centre = QPointF(self.width() / 2, 18 + radius + 8)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        turn = np.linspace(0, 2 * math.pi, 48)
        for points in ([self._project(math.cos(t), math.sin(t), 0, centre, radius) for t in turn],
                       [self._project(0, math.cos(t), math.sin(t), centre, radius) for t in turn]):
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.setPen(QPen(QColor(COLORS["border_soft"]), 1))
            painter.drawPath(path)
        painter.setFont(_font(MONO_FAMILY, 10))
        painter.setPen(QColor(COLORS["text_faint"]))
        marks = (("|0⟩", (0, 0, 1.18)), ("|1⟩", (0, 0, -1.25)), ("x", (1.2, 0, 0)), ("y", (0, 1.2, 0)))
        for text, point in marks:
            spot = self._project(*point, centre, radius)
            painter.drawText(QRectF(spot.x() - 14, spot.y() - 8, 28, 16),
                             int(Qt.AlignmentFlag.AlignCenter), text)
        x, y, z = self.vector
        tip = self._project(x, y, z, centre, radius)
        painter.setPen(QPen(QColor(SERIES[0]), 2))
        painter.drawLine(centre, tip)
        painter.setBrush(QColor(SERIES[0]))
        painter.setPen(QPen(QColor(COLORS["surface"]), 1.5))
        painter.drawEllipse(tip, 4, 4)
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 16), int(Qt.AlignmentFlag.AlignCenter), self.title)
        length = math.sqrt(x * x + y * y + z * z)
        painter.drawText(QRectF(0, self.height() - 16, self.width(), 16), int(Qt.AlignmentFlag.AlignCenter),
                         f"({x:+.2f}, {y:+.2f}, {z:+.2f}) |r|={length:.2f}")
