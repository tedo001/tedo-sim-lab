"""Quantum circuits: choose a gate, click a wire to place it (controls first, then the target),
and watch the exact state change — outcome probabilities, sampled shots, each qubit on its Bloch
sphere, and the amplitudes with their phases. Start from a textbook preset or an empty circuit."""

from __future__ import annotations

import cmath
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from labs.simulation.quantum import GATES, MAX_QUBITS, PRESETS, Circuit, Op, preset, purity, sample, simulate

from ...icons import icon
from ...widgets import Card, DataTable, FlowGrid, ResponsiveRow, label
from ...widgets.plots import BarList
from ...widgets.sim_views import BlochView
from .circuit_canvas import CircuitCanvas

__all__ = ["CircuitLab"]

PALETTE = ("H", "X", "Y", "Z", "S", "Sdg", "T", "Tdg", "SX", "RX", "RY", "RZ", "P", "U",
           "CX", "CZ", "CY", "CH", "CP", "CRZ", "SWAP", "CCX", "CSWAP")
_SHOWN = {"Sdg": "S†", "Tdg": "T†", "SX": "√X", "CX": "CNOT", "CCX": "Toffoli", "CSWAP": "Fredkin"}


class CircuitLab(QWidget):
    kind = "circuit"
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.circuit = preset("Bell state")
        self.history: list[list[Op]] = []
        self.selected = "H"
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        build = Card("Circuit", "pick a gate, then click a wire; right-click removes")
        row = QHBoxLayout()
        self.presets = QComboBox()
        self.presets.addItem("Presets…")
        self.presets.addItems(list(PRESETS))
        self.presets.activated.connect(self._preset_chosen)
        self.qubits = QSpinBox()
        self.qubits.setRange(1, MAX_QUBITS)
        self.qubits.setPrefix("qubits ")
        self.qubits.valueChanged.connect(self._resize)
        self.undo_button = QPushButton(icon("rotate-ccw"), "Undo")
        self.undo_button.clicked.connect(self.undo)
        self.clear_button = QPushButton(icon("eraser"), "Clear")
        self.clear_button.clicked.connect(self.clear)
        for widget in (self.presets, self.qubits, self.undo_button, self.clear_button):
            row.addWidget(widget)
        row.addStretch(1)
        build.add(row)
        self.description = label("", "CardCaption", wrap=True)
        build.add(self.description)

        self.gate_group = QButtonGroup(self)
        self.gate_buttons: dict[str, QToolButton] = {}
        for name in PALETTE:
            button = QToolButton()
            button.setText(_SHOWN.get(name, name))
            button.setCheckable(True)
            button.setFixedWidth(56)
            spec = GATES[name]
            button.setToolTip(f"{spec.title}" + (f": {spec.help}" if spec.help else ""))
            button.clicked.connect(lambda _=False, n=name: self.select_gate(n))
            self.gate_group.addButton(button)
            self.gate_buttons[name] = button
        self.palette_grid = FlowGrid(list(self.gate_buttons.values()), 56, fewest=6)
        build.add(self.palette_grid)
        angles = QHBoxLayout()
        self.angle_spins = []
        for name in ("θ", "φ", "λ"):
            spin = QDoubleSpinBox()
            spin.setRange(-4.0, 4.0)
            spin.setSingleStep(0.125)
            spin.setDecimals(3)
            spin.setValue(0.5)
            spin.setPrefix(f"{name} = ")
            spin.setSuffix(" π")
            angles.addWidget(spin)
            self.angle_spins.append(spin)
        self.instruction = label("", "CardCaption")
        angles.addWidget(self.instruction, 1)
        build.add(angles)
        self.canvas = CircuitCanvas()
        self.canvas.clicked.connect(self._clicked)
        self.canvas.remove_requested.connect(self._remove)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        build.add(scroll)
        self.canvas_scroll = scroll
        box.addWidget(build)

        probabilities = Card("Probabilities", "exact, before measuring")
        self.bars = BarList()
        probabilities.add(self.bars)

        shots = Card("Shots", "measuring every qubit, again and again")
        row = QHBoxLayout()
        self.shots = QSpinBox()
        self.shots.setRange(1, 1_000_000)
        self.shots.setValue(1024)
        self.shots.setPrefix("shots ")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setPrefix("seed ")
        for spin in (self.shots, self.seed):
            spin.valueChanged.connect(lambda _v: self.refresh())
            row.addWidget(spin)
        row.addStretch(1)
        shots.add(row)
        self.counts = BarList()
        shots.add(self.counts)
        box.addWidget(ResponsiveRow([(probabilities, 1), (shots, 1)], breakpoint=720))

        spheres = Card("Each qubit", "Bloch vectors; shorter than 1 = entangled with the others")
        self.sphere_row = QHBoxLayout()
        self.spheres: list[BlochView] = []
        spheres.add(self.sphere_row)
        box.addWidget(spheres)
        amplitudes = Card("State vector", "basis states with any weight")
        self.table = DataTable(("Basis", "Amplitude", "Probability", "Phase"), mono_columns=(0, 1, 2, 3),
                               stretch_column=1)
        amplitudes.add(self.table)
        box.addWidget(amplitudes)
        box.addStretch(1)
        self.select_gate("H")
        self.set_circuit(self.circuit)

    # Editing ------------------------------------------------------------------------------------------
    def select_gate(self, name: str) -> None:
        self.selected = name
        self.gate_buttons[name].setChecked(True)
        spec = GATES[name]
        for index, spin in enumerate(self.angle_spins):
            spin.setVisible(index < len(spec.params))
        self.canvas.set_pending(None, [])
        needed = spec.controls + spec.targets
        if needed == 1:
            text = f"Click a wire to place {_SHOWN.get(name, name)}."
        elif spec.controls:
            text = (f"Click the control{'s' if spec.controls > 1 else ''}, then the target"
                    f"{'s' if spec.targets > 1 else ''}.")
        else:
            text = "Click the two qubits to swap."
        self.instruction.setText(text)

    def _params(self) -> tuple[float, ...]:
        count = len(GATES[self.selected].params)
        return tuple(spin.value() * math.pi for spin in self.angle_spins[:count])

    def _clicked(self, column: int, qubit: int) -> None:
        spec = GATES[self.selected]
        needed = spec.controls + spec.targets
        pending, at = list(self.canvas.pending), self.canvas.pending_column
        if needed > 1:
            if at is None:
                self.canvas.set_pending(column, [qubit])
                return
            if qubit in pending:
                return
            pending.append(qubit)
            if len(pending) < needed:
                self.canvas.set_pending(at, pending)
                return
            column = at
        else:
            pending = [qubit]
        self.canvas.set_pending(None, [])
        targets, controls = tuple(pending[spec.controls:]), tuple(pending[:spec.controls])
        op = Op(self.selected, targets, controls, self._params(), column)
        self.place(op)

    def place(self, op: Op) -> bool:
        try:
            before = list(self.circuit.ops)
            self.circuit.add(op)
        except ValueError as exc:
            self.message.emit(str(exc))
            return False
        self.history.append(before)
        self._changed()
        return True

    def _remove(self, column: int, qubit: int) -> None:
        before = list(self.circuit.ops)
        if self.circuit.remove_at(column, qubit) is not None:
            self.history.append(before)
            self._changed()

    def undo(self) -> None:
        if self.history:
            self.circuit.ops = self.history.pop()
            self._changed()

    def clear(self) -> None:
        if self.circuit.ops:
            self.history.append(list(self.circuit.ops))
            self.circuit.ops = []
            self.circuit.name = ""
            self.description.setText("")
            self._changed()

    def _resize(self, qubits: int) -> None:
        if qubits != self.circuit.qubits:
            self.history.append(list(self.circuit.ops))
            self.circuit = self.circuit.resized(qubits)
            self._changed()

    def _preset_chosen(self, index: int) -> None:
        if index > 0:
            name = self.presets.itemText(index)
            self.history.append(list(self.circuit.ops))
            self.set_circuit(preset(name))
            self.presets.setCurrentIndex(0)

    def set_circuit(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.qubits.blockSignals(True)
        self.qubits.setValue(circuit.qubits)
        self.qubits.blockSignals(False)
        self.description.setText(PRESETS[circuit.name][0] if circuit.name in PRESETS else "")
        self._changed()

    def _changed(self) -> None:
        self.canvas.set_circuit(self.circuit)
        bar = self.canvas_scroll.horizontalScrollBar().sizeHint().height()
        self.canvas_scroll.setFixedHeight(self.canvas.sizeHint().height() + bar + 4)
        self.undo_button.setEnabled(bool(self.history))
        self.refresh()

    # Results ----------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        result = simulate(self.circuit)
        n = self.circuit.qubits
        amplitudes = result.amplitudes()
        top = sorted(amplitudes, key=lambda item: -item[2])
        self.bars.set_rows([(f"|{label}⟩", p) for label, _a, p in top], limit=16)
        counts = sample(result.state, self.shots.value(), self.seed.value())
        self.counts.set_rows([(f"|{label}⟩", count) for label, count in counts.items()], limit=16)
        while len(self.spheres) < n:
            view = BlochView()
            self.spheres.append(view)
            self.sphere_row.addWidget(view)
        for qubit, view in enumerate(self.spheres):
            view.setVisible(qubit < n)
            if qubit < n:
                mixed = purity(result.state, qubit, n)
                view.set_vector(result.bloch(qubit), f"q{qubit} · purity {mixed:.2f}")
        self.table.clear_rows()
        for label_text, amplitude, probability in amplitudes[:64]:
            phase = math.degrees(cmath.phase(amplitude)) if probability > 1e-12 else 0.0
            self.table.add_row([f"|{label_text}⟩", f"{amplitude.real:+.4f} {amplitude.imag:+.4f}i",
                                f"{probability:.4f}", f"{phase:+.1f}°"])
        if len(amplitudes) > 64:
            self.table.add_row(["…", f"{len(amplitudes) - 64} more basis states", "", ""])

    # Saving -------------------------------------------------------------------------------------------------
    def settings(self) -> dict:
        return {"circuit": self.circuit.to_dict(), "shots": self.shots.value(), "seed": self.seed.value()}

    def load(self, settings: dict) -> None:
        self.history.append(list(self.circuit.ops))
        self.shots.setValue(int(settings.get("shots", 1024)))
        self.seed.setValue(int(settings.get("seed", 0)))
        self.set_circuit(Circuit.from_dict(settings["circuit"]))

    def outputs(self) -> dict[str, bytes | str]:
        result = simulate(self.circuit)
        rows = ["basis,amplitude_real,amplitude_imag,probability"]
        rows += [f"{label},{a.real:.8f},{a.imag:.8f},{p:.8f}" for label, a, p in result.amplitudes()]
        return {"circuit.qasm": self.circuit.to_qasm(), "state.csv": "\n".join(rows) + "\n"}

    def default_name(self) -> str:
        return self.circuit.name or f"{self.circuit.qubits}-qubit circuit"
