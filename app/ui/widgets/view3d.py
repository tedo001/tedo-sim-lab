"""The camera shared by the 3-D views (a surface, a point cloud): drawn with QPainter, no OpenGL.

The viewer looks at the middle of the scene from an azimuth and an elevation, with a mild
perspective. Drag to turn, the wheel zooms, a double-click resets; ``camera_changed`` tells
when the person moved it, so two views can turn together. The scene's size on screen is fitted
to the widget when the data or the widget's size changes, never while turning, so it holds still.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["DEFAULT_CAMERA", "EYE", "OrbitView", "font"]

#: Azimuth and elevation (degrees) and zoom of a fresh view.
DEFAULT_CAMERA = (-35.0, 30.0, 1.0)
EYE = 3.4  # the viewer's distance from the middle, in scene units (the scene is about 2 across)


def font(family: str, size: int) -> QFont:
    face = QFont(family)
    face.setPixelSize(size)
    return face


class OrbitView(QWidget):
    #: The person turned or zoomed the view: azimuth, elevation, zoom.
    camera_changed = Signal(float, float, float)
    TOP, BOTTOM = 24, 44  # room for the title above and the legend below

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.azimuth, self.elevation, self.zoom = DEFAULT_CAMERA
        self._fit: float | None = None
        self._middle = (0.0, 0.0)  # the middle of the fitted scene on screen, in scene units
        self._drag: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(240, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("Drag to turn · wheel to zoom · double-click to reset")

    def sizeHint(self) -> QSize:
        return QSize(440, 400)

    # To give ----------------------------------------------------------------------------------------
    def corners(self) -> np.ndarray:
        """The scene's bounding box as points (N × 3); None-like empty when there is nothing."""
        return np.zeros((0, 3))

    def changed(self) -> None:
        """The picture is out of date (the camera moved, the data or the size changed)."""
        self.update()

    def hover_at(self, position: QPointF | None) -> None:
        """The pointer is over ``position`` (None: it left)."""

    # Camera -------------------------------------------------------------------------------------------
    def set_camera(self, azimuth: float, elevation: float, zoom: float, *, notify: bool = False) -> None:
        self.azimuth = azimuth % 360
        self.elevation = min(max(elevation, 5.0), 89.0)
        self.zoom = min(max(zoom, 0.4), 4.0)
        self.changed()
        if notify:
            self.camera_changed.emit(self.azimuth, self.elevation, self.zoom)

    def refit(self) -> None:
        """Fit the scene to the widget again (after new data or a new size)."""
        self._fit = None
        self.changed()

    def _rotate(self, u, v, w) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        across = u * math.cos(a) - v * math.sin(a)
        ahead = u * math.sin(a) + v * math.cos(a)
        return across, w * math.cos(e) + ahead * math.sin(e), ahead * math.cos(e) - w * math.sin(e)

    def _fitted(self) -> float:
        if self._fit is None:
            corners = self.corners()
            if not len(corners):
                return 1.0
            across, up, depth = self._rotate(corners[:, 0], corners[:, 1], corners[:, 2])
            perspective = EYE / (EYE + depth)
            flat_x, flat_y = across * perspective, up * perspective
            width = np.ptp(flat_x) or 1.0
            height = np.ptp(flat_y) or 1.0
            room_w, room_h = max(self.width() - 24, 40), max(self.height() - self.TOP - self.BOTTOM - 8, 40)
            self._fit = min(room_w / width, room_h / height)
            self._middle = ((flat_x.max() + flat_x.min()) / 2, (flat_y.max() + flat_y.min()) / 2)
        return self._fit

    def project(self, u, v, w) -> tuple[np.ndarray, np.ndarray]:
        """Scene points → (screen points … × 2, depth: larger is farther)."""
        across, up, depth = self._rotate(np.asarray(u, float), np.asarray(v, float), np.asarray(w, float))
        perspective = EYE / (EYE + depth)
        scale = self._fitted() * self.zoom
        centre_x = self.width() / 2
        centre_y = self.TOP + (self.height() - self.TOP - self.BOTTOM) / 2
        middle_x, middle_y = self._middle
        return np.stack([centre_x + (across * perspective - middle_x) * scale,
                         centre_y - (up * perspective - middle_y) * scale], axis=-1), depth

    def pixels(self, depth) -> np.ndarray:
        """Pixels per scene unit at a depth."""
        return EYE / (EYE + np.asarray(depth, float)) * self._fitted() * self.zoom

    def eye(self) -> np.ndarray:
        """Where the viewer stands, in scene units."""
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        ahead = -EYE * math.cos(e)
        return np.array([ahead * math.sin(a), ahead * math.cos(a), EYE * math.sin(e)])

    # Interaction -------------------------------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self.refit()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag
            self._drag = event.position()
            self.set_camera(self.azimuth - delta.x() * 0.5, self.elevation + delta.y() * 0.4, self.zoom,
                            notify=True)
            return
        self.hover_at(event.position())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._drag = None

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.hover_at(None)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt API)
        turns = event.angleDelta().y() / 120
        self.set_camera(self.azimuth, self.elevation, self.zoom * 1.12 ** turns, notify=True)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.set_camera(*DEFAULT_CAMERA[:2], DEFAULT_CAMERA[2], notify=True)
        self.refit()
