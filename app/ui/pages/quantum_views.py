"""Views of a trained quantum classifier: its decision boundary over the angle plane (with the test
points), the circuit with its trained angles, and the quantum kernel between training samples."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme.tokens import COLORS, MONO_FAMILY

__all__ = ["BoundaryView", "CircuitDiagram", "KernelView"]

CLASS_COLOURS = ("series_1", "series_2", "series_3")


def _colour(index: int) -> QColor:
    return QColor(COLORS[CLASS_COLOURS[index]] if index < len(CLASS_COLOURS) else COLORS["series_other"])


class BoundaryView(QWidget):
    """The predicted class over the plane of the two angles (0…π each), shaded by confidence; the
    test points on top, ringed in the surface colour."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.boundary: dict[str, Any] | None = None
        self.classes: list[str] = []
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(320, 300)

    def set_boundary(self, boundary: dict[str, Any] | None, classes: list[str]) -> None:
        self.boundary, self.classes = boundary, classes
        self.setFixedHeight(300 if boundary else 0)
        self.update()

    def _image(self) -> QImage:
        grid = np.asarray(self.boundary["classes"])
        confidence = np.asarray(self.boundary.get("confidence") or np.ones_like(grid, dtype=float))
        size = grid.shape[0]
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        surface = QColor(COLORS["surface"])
        for row in range(size):
            for col in range(size):
                colour = _colour(int(grid[row, col]))
                share = 0.18 + 0.4 * max(0.0, (float(confidence[row, col]) - 0.5) * 2)
                mixed = QColor(int(surface.red() + (colour.red() - surface.red()) * share),
                               int(surface.green() + (colour.green() - surface.green()) * share),
                               int(surface.blue() + (colour.blue() - surface.blue()) * share))
                image.setPixelColor(col, size - 1 - row, mixed)  # angle 2 grows upwards
        return image

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 (Qt)
        if not self.boundary:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width() - 40, self.height() - 34)
        plot = QRectF(32, 6, side, side)
        painter.drawImage(plot, self._image())
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRect(plot)
        for x1, x2, label in self.boundary.get("points", []):
            centre = QPointF(plot.left() + float(x1) / np.pi * plot.width(),
                             plot.bottom() - float(x2) / np.pi * plot.height())
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))
            painter.setBrush(_colour(int(label)))
            painter.drawEllipse(centre, 4, 4)
        painter.setPen(QColor(COLORS["text_dim"]))
        font = QFont(MONO_FAMILY)
        font.setPixelSize(10)
        painter.setFont(font)
        axes = self.boundary.get("axes", ["angle 1", "angle 2"])
        painter.drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 14),
                         Qt.AlignmentFlag.AlignCenter,
                         f"{axes[0]} → 0 … π")
        painter.save()
        painter.translate(plot.left() - 8, plot.bottom())
        painter.rotate(-90)
        painter.drawText(QRectF(0, -12, plot.height(), 14), Qt.AlignmentFlag.AlignCenter,
                         f"{axes[1]} → 0 … π")
        painter.restore()
        painter.end()


class CircuitDiagram(QWidget):
    """Wires, gates and their angles: data angles by name (x1, x2, …), trained ones as numbers."""

    CELL, ROW, LEFT = 58, 34, 36

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.qubits = 0
        self.placed: list[tuple[int, str, list[int], str]] = []  # column, gate, qubits, text

    def set_circuit(self, circuit: dict[str, Any] | None, params: np.ndarray | None,
                    names: list[str]) -> None:
        self.placed, self.qubits = [], int(circuit["qubits"]) if circuit else 0
        busy = [0] * self.qubits
        for gate, qubits, angle in (circuit or {}).get("gates", []):
            low, high = min(qubits), max(qubits)
            column = max(busy[low:high + 1])
            for q in range(low, high + 1):
                busy[q] = column + 1
            text = ""
            if angle.startswith("θ") and params is not None and int(angle[1:]) < len(params):
                text = f"{float(params[int(angle[1:])]):.2f}"
            elif angle:
                parts = [int(p) for p in angle[1:].split("x")]
                text = "·".join(names[p] if p < len(names) else f"x{p}" for p in parts)
            self.placed.append((column, gate, list(qubits), text))
        columns = max(busy, default=0)
        self.setFixedSize(self.LEFT + columns * self.CELL + 16, 12 + self.qubits * self.ROW)
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 (Qt)
        if not self.qubits:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(MONO_FAMILY)
        font.setPixelSize(10)
        painter.setFont(font)
        wire = QPen(QColor(COLORS["border"]), 1)
        for q in range(self.qubits):
            y = 6 + q * self.ROW + self.ROW / 2
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(QRectF(0, y - 8, self.LEFT - 6, 16), Qt.AlignmentFlag.AlignRight, f"q{q}")
            painter.setPen(wire)
            painter.drawLine(QPointF(self.LEFT, y), QPointF(self.width() - 6, y))
        for column, gate, qubits, text in self.placed:
            x = self.LEFT + column * self.CELL + self.CELL / 2
            ys = [6 + q * self.ROW + self.ROW / 2 for q in qubits]
            if gate == "CNOT":
                painter.setPen(QPen(QColor(COLORS["accent"]), 2))
                painter.drawLine(QPointF(x, ys[0]), QPointF(x, ys[1]))
                painter.setBrush(QColor(COLORS["accent"]))
                painter.drawEllipse(QPointF(x, ys[0]), 4, 4)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(x, ys[1]), 8, 8)
                painter.drawLine(QPointF(x - 8, ys[1]), QPointF(x + 8, ys[1]))
                painter.drawLine(QPointF(x, ys[1] - 8), QPointF(x, ys[1] + 8))
                continue
            box = QRectF(x - self.CELL / 2 + 4, ys[0] - 12, self.CELL - 8, 24)
            painter.setPen(QPen(QColor(COLORS["border"]), 1))
            painter.setBrush(QColor(COLORS["raised"]))
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(box.adjusted(0, 1, 0, -11), Qt.AlignmentFlag.AlignCenter, gate)
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(box.adjusted(0, 11, 0, 0), Qt.AlignmentFlag.AlignCenter, text[:8])
        painter.end()


class KernelView(QWidget):
    """The kernel matrix |⟨φ(x)|φ(x')⟩|² between training samples (sorted by class), one hue from
    dark (0) to light (1)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.matrix: np.ndarray | None = None
        self.setMinimumSize(200, 200)

    def sizeHint(self) -> QSize:
        return QSize(260, 260)

    def set_matrix(self, matrix: Any) -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=float)
        self.setFixedHeight(260 if self.matrix is not None else 0)
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 (Qt)
        if self.matrix is None or not self.matrix.size:
            return
        low, high = QColor(COLORS["seq_low"]), QColor(COLORS["seq_high"])
        n = self.matrix.shape[0]
        image = QImage(n, n, QImage.Format.Format_RGB32)
        for i in range(n):
            for j in range(n):
                t = float(np.clip(self.matrix[i, j], 0, 1))
                image.setPixelColor(j, i, QColor(int(low.red() + (high.red() - low.red()) * t),
                                                 int(low.green() + (high.green() - low.green()) * t),
                                                 int(low.blue() + (high.blue() - low.blue()) * t)))
        side = min(self.width(), self.height()) - 8
        painter = QPainter(self)
        painter.drawImage(QRectF(4, 4, side, side), image)
        painter.end()
