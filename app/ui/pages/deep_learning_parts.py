"""The Deep Learning page's parts: the layer list editor and the stack diagram."""

from __future__ import annotations

import copy
import math
from typing import Any

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labs.deep_learning.layers import KINDS, Step, starter_stack

from ..icons import icon
from ..theme import mono_font
from ..theme.tokens import COLORS
from ..widgets import label

__all__ = ["LayerEditor", "StackDiagram"]

#: Lowest value each integer setting accepts (the planner checks the rest).
_LOW = {"padding": 0}


class LayerEditor(QWidget):
    """A list of layers with add / remove / move and a form for the selected layer's settings."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layers: list[dict[str, Any]] = starter_stack()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        add_row = QHBoxLayout()
        self.kind = QComboBox()
        for kind, (title, _defaults, help_text) in KINDS.items():
            self.kind.addItem(f"{title} — {help_text}", kind)
        add_row.addWidget(self.kind, 1)
        self.add_button = QPushButton(icon("plus"), "Add")
        self.add_button.clicked.connect(self.add_selected_kind)
        add_row.addWidget(self.add_button)
        box.addLayout(add_row)

        self.list = QListWidget()
        self.list.setFont(mono_font())
        self.list.setMinimumHeight(180)
        self.list.currentRowChanged.connect(self._show_settings)
        box.addWidget(self.list)

        buttons = QHBoxLayout()
        self.up = QPushButton(icon("arrow-up"), "Up")
        self.down = QPushButton(icon("arrow-down"), "Down")
        self.remove = QPushButton(icon("trash-2"), "Remove")
        self.reset = QPushButton("Starter stack")
        self.clear = QPushButton("Clear")
        self.up.clicked.connect(lambda: self.move(-1))
        self.down.clicked.connect(lambda: self.move(1))
        self.remove.clicked.connect(self.remove_selected)
        self.reset.clicked.connect(lambda: self.set_layers(starter_stack()))
        self.clear.clicked.connect(lambda: self.set_layers([]))
        for button in (self.up, self.down, self.remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.reset)
        buttons.addWidget(self.clear)
        box.addLayout(buttons)

        self.settings_title = label("", "BodyStrong")
        box.addWidget(self.settings_title)
        self.form_holder = QWidget()
        self.form = QFormLayout(self.form_holder)
        self.form.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.form_holder)
        self._fill(0)

    # Editing --------------------------------------------------------------------------------
    def set_layers(self, layers: list[dict[str, Any]]) -> None:
        self.layers = copy.deepcopy(list(layers))
        self._fill(0 if self.layers else -1)
        self.changed.emit()

    def add(self, kind: str) -> None:
        row = self.list.currentRow()
        at = row + 1 if row >= 0 else len(self.layers)
        self.layers.insert(at, {"type": kind, **KINDS[kind][1]})
        self._fill(at)
        self.changed.emit()

    def add_selected_kind(self) -> None:
        self.add(self.kind.currentData())

    def remove_selected(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.layers):
            del self.layers[row]
            self._fill(min(row, len(self.layers) - 1))
            self.changed.emit()

    def move(self, by: int) -> None:
        row = self.list.currentRow()
        target = row + by
        if 0 <= row < len(self.layers) and 0 <= target < len(self.layers):
            self.layers[row], self.layers[target] = self.layers[target], self.layers[row]
            self._fill(target)
            self.changed.emit()

    def set_value(self, row: int, key: str, value: Any) -> None:
        self.layers[row][key] = value
        self._relabel(row)
        self.changed.emit()

    # Showing --------------------------------------------------------------------------------
    @staticmethod
    def describe(layer: dict[str, Any]) -> str:
        title = KINDS[layer["type"]][0]
        inside = ", ".join(f"{k}={v}" for k, v in layer.items() if k != "type")
        return f"{title}({inside})" if inside else title

    def _relabel(self, row: int) -> None:
        item = self.list.item(row)
        if item is not None:
            item.setText(f"{row + 1:>2}  {self.describe(self.layers[row])}")

    def _fill(self, select: int = -1) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for row in range(len(self.layers)):
            self.list.addItem("")
            self._relabel(row)
        self.list.blockSignals(False)
        self.list.setCurrentRow(select)
        self._show_settings(select)

    def _show_settings(self, row: int) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        valid = 0 <= row < len(self.layers)
        for button in (self.remove, self.up, self.down):
            button.setEnabled(valid)
        if not valid:
            self.settings_title.setText("No layer selected" if self.layers else
                                        "No layers: the network is just the final Linear layer")
            return
        layer = self.layers[row]
        defaults = KINDS[layer["type"]][1]
        self.settings_title.setText(f"Layer {row + 1}: {KINDS[layer['type']][0]}"
                                    + ("" if defaults else " (no settings)"))
        for key, default in defaults.items():
            value = layer.get(key, default)
            if isinstance(default, float):
                field = QDoubleSpinBox()
                field.setRange(0.0, 0.95)
                field.setSingleStep(0.05)
                field.setValue(float(value))
                field.valueChanged.connect(lambda v, r=row, k=key: self.set_value(r, k, round(v, 4)))
            else:
                field = QSpinBox()
                field.setRange(_LOW.get(key, 1), 4096)
                field.setValue(int(value))
                field.valueChanged.connect(lambda v, r=row, k=key: self.set_value(r, k, int(v)))
            field.setObjectName(f"setting_{key}")
            self.form.addRow(key, field)


class StackDiagram(QWidget):
    """The planned stack left to right: one block per layer, its height following the size of
    the maps (vectors are thin bars), the output shape under it. Automatic layers are dashed."""

    BLOCK = 74
    GAP = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.steps: list[Step] = []
        self.input_shape: tuple[int, ...] = ()
        self.setMinimumHeight(170)

    def set_steps(self, steps: list[Step], input_shape: tuple[int, ...]) -> None:
        self.steps, self.input_shape = list(steps), tuple(input_shape)
        self.setMinimumWidth((len(self.steps) + 1) * (self.BLOCK + self.GAP) + 16)
        self.update()

    def _height(self, shape: tuple[int, ...], largest: float, room: float) -> float:
        if len(shape) == 3:
            return max(18.0, room * math.sqrt(shape[1] * shape[2]) / largest)
        return 12.0

    def paintEvent(self, _event: Any) -> None:
        if not self.input_shape:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(mono_font(11))
        items = [("Input", self.input_shape, False), *((s.title, s.shape, s.automatic) for s in self.steps)]
        largest = max(math.sqrt(sh[1] * sh[2]) for _t, sh, _a in items if len(sh) == 3)
        room, middle = self.height() - 64.0, (self.height() - 40) / 2 + 4
        for index, (title, shape, automatic) in enumerate(items):
            x = 8 + index * (self.BLOCK + self.GAP)
            height = self._height(shape, largest, room)
            rect = QRectF(x, middle - height / 2, self.BLOCK, height)
            pen = QPen(QColor(COLORS["text_faint"] if automatic else COLORS["border"]), 1)
            if automatic:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(COLORS["selected"] if index == 0 else COLORS["raised"]))
            painter.drawRoundedRect(rect, 4, 4)
            if index:
                painter.setPen(QPen(QColor(COLORS["text_faint"]), 1))
                painter.drawLine(int(x - self.GAP + 2), int(middle), int(x - 2), int(middle))
            painter.setPen(QColor(COLORS["text"] if not automatic else COLORS["text_dim"]))
            painter.drawText(QRectF(x - 4, 2, self.BLOCK + 8, 18), Qt.AlignmentFlag.AlignCenter, title)
            painter.setPen(QColor(COLORS["text_dim"]))
            flags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
            painter.drawText(QRectF(x - 4, self.height() - 34, self.BLOCK + 8, 30), flags,
                             "×".join(map(str, shape)))
        painter.end()
