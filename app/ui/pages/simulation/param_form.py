"""A form built from an operation's parameter specs: number boxes with their ranges, choices,
switches, and a 3×3 grid for a convolution kernel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QSpinBox,
    QWidget,
)

from labs.simulation.vision.ops import Param

from ...widgets import label

__all__ = ["ParamForm"]


class ParamForm(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.params: tuple[Param, ...] = ()
        self.editors: dict[str, object] = {}

    def set_params(self, params: tuple[Param, ...], values: dict[str, object]) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.params, self.editors = params, {}
        if not params:
            self.form.addRow(label("No settings.", "CardCaption"))
            return
        for param in params:
            value = param.clean(values.get(param.name, param.default))
            editor = self._editor(param, value)
            self.editors[param.name] = editor
            self.form.addRow(param.title, editor)

    def _editor(self, param: Param, value):
        if param.kind == "float":
            spin = QDoubleSpinBox()
            spin.setRange(param.low, param.high)
            spin.setSingleStep(param.step)
            spin.setDecimals(3 if param.step < 0.01 else 2)
            spin.setValue(float(value))
            spin.valueChanged.connect(lambda _v: self.changed.emit())
            return spin
        if param.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(param.low), int(param.high))
            spin.setSingleStep(int(param.step) or 1)
            spin.setValue(int(value))
            spin.valueChanged.connect(lambda _v: self.changed.emit())
            return spin
        if param.kind == "choice":
            combo = QComboBox()
            combo.addItems(list(param.choices))
            combo.setCurrentText(str(value))
            combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
            return combo
        if param.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda _on: self.changed.emit())
            return box
        grid = QWidget()
        layout = QGridLayout(grid)
        layout.setContentsMargins(0, 0, 0, 0)
        spins = []
        for index, number in enumerate(value):
            spin = QDoubleSpinBox()
            spin.setRange(-50, 50)
            spin.setSingleStep(0.25)
            spin.setDecimals(2)
            spin.setValue(float(number))
            spin.valueChanged.connect(lambda _v: self.changed.emit())
            layout.addWidget(spin, index // 3, index % 3)
            spins.append(spin)
        grid.spins = spins
        return grid

    def values(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for param in self.params:
            editor = self.editors[param.name]
            if param.kind in ("float", "int"):
                out[param.name] = editor.value()
            elif param.kind == "choice":
                out[param.name] = editor.currentText()
            elif param.kind == "bool":
                out[param.name] = editor.isChecked()
            else:
                out[param.name] = tuple(spin.value() for spin in editor.spins)
        return out
