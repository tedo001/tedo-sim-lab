"""The camera model: move and turn a pinhole camera around a cube, a ground grid and the world
axes; change its focal length, principal point and lens distortion, and see the image it takes
together with its matrices K, [R | t] and P = K [R | t]."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from labs.simulation.vision import Camera, projection_matrix, wireframe
from labs.simulation.vision.camera import OBJECTS, intrinsics, rotation

from ...icons import icon
from ...theme import mono_font
from ...theme.tokens import COLORS, MONO_FAMILY, SERIES
from ...widgets import Card, ResponsiveRow
from ...widgets.plots import _font

__all__ = ["CameraLab", "CameraView"]

COLOURS = {"Cube": SERIES[0], "Ground grid": COLORS["series_other"], "Axes": SERIES[1]}


class CameraView(QWidget):
    """The camera's image: the sensor's rectangle with the projected wireframe inside it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera = Camera()
        self.edges: list = []
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(560, 420)

    def show_camera(self, camera: Camera, objects: tuple[str, ...]) -> None:
        self.camera, self.edges = camera, wireframe(camera, objects, samples=32)
        self.update()

    def sensor(self) -> QRectF:
        cam = self.camera
        scale = min((self.width() - 20) / cam.width, (self.height() - 30) / cam.height)
        w, h = cam.width * scale, cam.height * scale
        return QRectF((self.width() - w) / 2, 20 + (self.height() - 30 - h) / 2, w, h)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.sensor()
        painter.fillRect(rect, QColor(COLORS["bg"]))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRect(rect)
        scale = rect.width() / self.camera.width
        painter.setClipRect(rect)
        for name, uv, visible in self.edges:
            path, drawing = QPainterPath(), False
            for (u, v), seen in zip(uv, visible, strict=True):
                point = QPointF(rect.left() + u * scale, rect.top() + v * scale)
                if seen and drawing:
                    path.lineTo(point)
                elif seen:
                    path.moveTo(point)
                drawing = bool(seen)
            painter.strokePath(path, QPen(QColor(COLOURS[name]), 2 if name != "Ground grid" else 1))
        painter.setClipping(False)
        centre = QPointF(rect.left() + self.camera.cx * scale, rect.top() + self.camera.cy * scale)
        painter.setPen(QPen(QColor(COLORS["text_faint"]), 1))
        painter.drawLine(QPointF(centre.x() - 6, centre.y()), QPointF(centre.x() + 6, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - 6), QPointF(centre.x(), centre.y() + 6))
        painter.setFont(_font(MONO_FAMILY, 11))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, 0, self.width(), 18), int(Qt.AlignmentFlag.AlignLeft),
                         f"image {self.camera.width} × {self.camera.height} px · + principal point")


def _spin(low: float, high: float, step: float, value: float, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(3 if step < 0.01 else 2 if step < 1 else 1)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


class CameraLab(QWidget):
    kind = "camera"
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        controls = Card("Camera", "angles in degrees, distances in scene units")
        form = QFormLayout()
        default = Camera()
        self.spins = {
            "yaw": _spin(-180, 180, 1, default.yaw, "°"), "pitch": _spin(-89, 89, 1, default.pitch, "°"),
            "roll": _spin(-180, 180, 1, default.roll, "°"), "tx": _spin(-20, 20, 0.1, default.tx),
            "ty": _spin(-20, 20, 0.1, default.ty), "tz": _spin(0.1, 50, 0.1, default.tz),
            "focal": _spin(50, 3000, 10, default.focal, " px"), "cx": _spin(0, 2000, 1, default.cx, " px"),
            "cy": _spin(0, 2000, 1, default.cy, " px"), "k1": _spin(-1, 1, 0.01, default.k1),
            "k2": _spin(-1, 1, 0.01, default.k2),
        }
        titles = {"yaw": "Yaw", "pitch": "Pitch", "roll": "Roll", "tx": "t x", "ty": "t y",
                  "tz": "t z (distance)",
                  "focal": "Focal length f", "cx": "Principal point cx", "cy": "Principal point cy",
                  "k1": "Distortion k₁", "k2": "Distortion k₂"}
        for key, spin in self.spins.items():
            form.addRow(titles[key], spin)
            spin.valueChanged.connect(lambda _v: self.refresh())
        controls.add(form)
        self.object_boxes = {name: QCheckBox(name) for name in OBJECTS}
        objects = QHBoxLayout()
        for widget in self.object_boxes.values():
            widget.setChecked(True)
            widget.toggled.connect(lambda _on: self.refresh())
            objects.addWidget(widget)
        controls.add(objects)
        reset = QPushButton(icon("rotate-ccw"), "Reset the camera")
        reset.clicked.connect(lambda: self.load({"camera": Camera().to_dict()}))
        controls.add(reset)
        image = Card("What the camera sees", "")
        self.view = CameraView()
        image.add(self.view)
        box.addWidget(ResponsiveRow([(controls, 0), (image, 1)], breakpoint=760))
        matrices = Card("Matrices", "x ~ P X with P = K [R | t]; distortion is applied after projection")
        self.matrices = QPlainTextEdit()
        self.matrices.setObjectName("Code")
        self.matrices.setReadOnly(True)
        self.matrices.setFont(mono_font())
        matrices.add(self.matrices)
        box.addWidget(matrices)
        box.addStretch(1)
        self.refresh()

    def camera(self) -> Camera:
        values = {key: spin.value() for key, spin in self.spins.items()}
        return Camera(**values)

    def refresh(self) -> None:
        camera = self.camera()
        objects = tuple(name for name, widget in self.object_boxes.items() if widget.isChecked())
        self.view.show_camera(camera, objects)
        rotation_matrix = rotation(camera.yaw, camera.pitch, camera.roll)
        translation = np.array([camera.tx, camera.ty, camera.tz])
        with np.printoptions(precision=3, suppress=True):
            text = (f"K =\n{intrinsics(camera)}\n\nR =\n{rotation_matrix}\n\nt = {translation}"
                    f"\n\nP = K [R | t] =\n{projection_matrix(camera)}"
                    f"\n\ndistortion d(r) = 1 + {camera.k1:g}·r² + {camera.k2:g}·r⁴")
        self.matrices.setPlainText(text)
        lines = text.count("\n") + 1  # tall enough to show it all
        self.matrices.setFixedHeight(lines * self.matrices.fontMetrics().lineSpacing() + 18)

    def settings(self) -> dict:
        objects = [name for name, widget in self.object_boxes.items() if widget.isChecked()]
        return {"camera": self.camera().to_dict(), "objects": objects}

    def load(self, settings: dict) -> None:
        camera = Camera.from_dict(settings.get("camera", {}))
        for key, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(float(getattr(camera, key)))
            spin.blockSignals(False)
        for name, widget in self.object_boxes.items():
            widget.blockSignals(True)
            widget.setChecked(name in settings.get("objects", list(OBJECTS)))
            widget.blockSignals(False)
        self.refresh()

    def outputs(self) -> dict[str, bytes | str]:
        return {"matrices.txt": self.matrices.toPlainText() + "\n"}

    def default_name(self) -> str:
        return "Camera model"
