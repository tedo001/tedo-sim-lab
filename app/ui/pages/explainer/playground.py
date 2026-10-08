"""CNN Explainer's hyperparameter view: change input size, padding, kernel size and
stride, and watch where the kernel lands and how big the output becomes."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labs.computer_vision.explainer import ConvGeometry

from ...icons import icon
from ...theme.tokens import COLORS
from ...widgets import ResponsiveRow, label
from .colours import mono_font

__all__ = ["GeometryView", "Playground"]

_STEP_MS = 600


class GeometryView(QWidget):
    """The padded input (padding cells hatched) with the kernel outlined, and the output with
    the matching cell filled. Cells the kernel never reaches are struck through."""

    hovered = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.geometry_ = ConvGeometry(5, 3)
        self.position = (0, 0)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(180)

    def set_geometry(self, geometry: ConvGeometry, position: tuple[int, int]) -> None:
        self.geometry_, self.position = geometry, position
        self.setFixedHeight(int(self._cell() * max(geometry.padded, 1) + 24))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(420, self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.setFixedHeight(int(self._cell() * max(self.geometry_.padded, 1) + 24))
        super().resizeEvent(event)

    def _cell(self) -> float:
        g = self.geometry_
        return max(min((self.width() - 60) / (g.padded + g.output), 30.0), 10.0)

    def _origins(self) -> tuple[float, float]:
        cell = self._cell()
        return 0.0, self.geometry_.padded * cell + 40

    def paintEvent(self, event: QPaintEvent) -> None:
        g, cell = self.geometry_, self._cell()
        left, right = self._origins()
        painter = QPainter(self)
        painter.setFont(mono_font(10))
        covered = g.coverage()
        for row in range(g.padded):
            for column in range(g.padded):
                rect = QRectF(left + column * cell, 20 + row * cell, cell, cell)
                padding = g.is_padding(row, column)
                painter.fillRect(rect, QColor(COLORS["bg"] if padding else COLORS["raised"]))
                style = Qt.PenStyle.DashLine if padding else Qt.PenStyle.SolidLine
                painter.setPen(QPen(QColor(COLORS["border"]), 1, style))
                painter.drawRect(rect)
                if not covered[row, column]:
                    painter.setPen(QPen(QColor(COLORS["warn"]), 1))
                    painter.drawLine(rect.topLeft(), rect.bottomRight())
        top, start = g.window(*self.position)
        painter.setPen(QPen(QColor(COLORS["text"]), 2))
        painter.drawRect(QRectF(left + start * cell, 20 + top * cell, g.kernel * cell, g.kernel * cell))
        for row in range(g.output):
            for column in range(g.output):
                rect = QRectF(right + column * cell, 20 + row * cell, cell, cell)
                current = (row, column) == self.position
                painter.fillRect(rect, QColor(COLORS["series_1"] if current else COLORS["raised"]))
                painter.setPen(QPen(QColor(COLORS["border"]), 1))
                painter.drawRect(rect)
        painter.setPen(QColor(COLORS["text_faint"]))
        painter.drawText(QRectF(left, 0, g.padded * cell + 30, 16), int(Qt.AlignmentFlag.AlignLeft),
                         f"input {g.input}×{g.input}" + (f" + padding {g.padding}" if g.padding else ""))
        painter.drawText(QRectF(right, 0, g.output * cell + 60, 16), int(Qt.AlignmentFlag.AlignLeft),
                         f"output {g.output}×{g.output}")
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        g, cell = self.geometry_, self._cell()
        left, right = self._origins()
        point = event.position()
        row = int((point.y() - 20) // cell)
        if point.x() >= right:
            column = int((point.x() - right) // cell)
            if 0 <= row < g.output and 0 <= column < g.output:
                self.hovered.emit(row, column)
        elif 0 <= row < g.padded:
            column = int((point.x() - left) // cell)
            if 0 <= column < g.padded:
                self.hovered.emit(min(row // g.stride, g.output - 1), min(column // g.stride, g.output - 1))


class Playground(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(label(
            "Padding adds a border of zeros so the kernel can centre on edge pixels and the map keeps "
            "its size. A larger kernel sees more at once but shrinks the map faster. Stride is how far "
            "the kernel moves each step: a larger stride gives a smaller output. When the stride does "
            "not fit evenly, PyTorch drops the cells the kernel cannot reach (struck through).",
            "Body", wrap=True))
        self.spins: dict[str, QSpinBox] = {}
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        for key, title in (("input", "Input size"), ("padding", "Padding"), ("kernel", "Kernel size"),
                           ("stride", "Stride")):
            spin = QSpinBox()
            spin.setRange(0, 15)
            spin.valueChanged.connect(self._changed)
            self.spins[key] = spin
            form.addRow(title, spin)
        self.formula = label("", "Mono", wrap=True)
        self.note = label("", "CardCaption", wrap=True)
        left = QWidget()
        box = QVBoxLayout(left)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(form)
        box.addWidget(label("output size", "CardCaption"))
        box.addWidget(self.formula)
        box.addWidget(self.note)
        box.addStretch(1)
        self.view = GeometryView()
        self.view.hovered.connect(self._hovered)
        layout.addWidget(ResponsiveRow([(left, 0), (self.view, 1)], breakpoint=620))
        self.play_button = QPushButton()
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggled)
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(label("Hover the grids to move the kernel.", "CardCaption"), 1)
        layout.addLayout(controls)
        self.timer = QTimer(self)
        self.timer.setInterval(_STEP_MS)
        self.timer.timeout.connect(self._advance)
        self.geometry_ = ConvGeometry(5, 3)
        self.position = (0, 0)
        self._updating = False
        self.set_values(input=5, kernel=3, padding=0, stride=1)
        self._toggled(False)

    def set_values(self, **values: int) -> None:
        current = {key: spin.value() for key, spin in self.spins.items()} | values
        self.geometry_ = ConvGeometry.fitted(current["input"], current["kernel"], current["padding"],
                                             current["stride"])
        g = self.geometry_
        self._updating = True
        for key, low, high in (("input", 3, 9), ("padding", 0, g.max_padding),
                               ("kernel", 1, min(g.max_kernel, 9)),
                               ("stride", 1, g.max_stride)):
            spin = self.spins[key]
            spin.setRange(low, high)
            spin.setValue(getattr(g, key))
        self._updating = False
        self.position = (min(self.position[0], g.output - 1), min(self.position[1], g.output - 1))
        self.formula.setText(g.formula())
        self.note.setText(f"The kernel never reaches the last {g.leftover} row(s) and column(s)."
                          if g.leftover else "")
        self.view.set_geometry(g, self.position)

    def _changed(self) -> None:
        if not self._updating:
            self.set_values()

    def _advance(self) -> None:
        g = self.geometry_
        flat = (self.position[0] * g.output + self.position[1] + 1) % (g.output * g.output)
        self.position = divmod(flat, g.output)
        self.view.set_geometry(g, self.position)

    def _hovered(self, y: int, x: int) -> None:
        self.play_button.setChecked(False)
        self.position = (y, x)
        self.view.set_geometry(self.geometry_, self.position)

    def _toggled(self, on: bool) -> None:
        self.play_button.setIcon(icon("pause" if on else "play"))
        self.play_button.setText("Pause" if on else "Play")
        if on:
            self.timer.start()
        else:
            self.timer.stop()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.play_button.setChecked(False)
        super().hideEvent(event)
