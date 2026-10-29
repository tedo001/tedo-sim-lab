"""The circuit tool's device switch: *Ideal simulation* (the exact state vector) or *Noisy hardware
simulation* (a density matrix with depolarizing errors after one- and two-qubit gates, T1/T2
relaxation for each gate's duration and readout bit flips, as in :mod:`labs.simulation.quantum.noise`).
It is a simulation of an imperfect device, not a device."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QWidget

from labs.simulation.quantum.noise import REFERENCE, NoiseModel

from ...widgets import Card, Segmented, label

__all__ = ["NoiseControls"]


def _box(low: float, high: float, value: float, decimals: int, step: float, tip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setMaximumWidth(160)
    box.setToolTip(tip)
    return box


class NoiseControls(Card):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Device", "an ideal simulation, or a simulated imperfect device", parent=parent)
        self.mode = Segmented((("ideal", "Ideal simulation"), ("noisy", "Noisy hardware simulation")))
        self.mode.changed.connect(lambda _key: self._mode_changed())
        self.add(self.mode)
        self.fields = QWidget()
        rows = QFormLayout(self.fields)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        ref = REFERENCE
        self.p1 = _box(0, 1, ref.depolarizing_1q, 4, 0.001, "Depolarizing λ after every one-qubit gate")
        self.p2 = _box(0, 1, ref.depolarizing_2q, 4, 0.005,
                       "Depolarizing λ after every gate on two or more qubits")
        self.t1 = _box(0, 1e6, ref.t1_us, 1, 10, "Relaxation time T1 in µs; 0 = no relaxation")
        self.t2 = _box(0, 2e6, ref.t2_us, 1, 10, "Coherence time T2 in µs, at most 2 × T1")
        self.g1 = _box(1, 1e5, ref.gate_time_1q_ns, 0, 10, "Duration of a one-qubit gate (ns)")
        self.g2 = _box(1, 1e5, ref.gate_time_2q_ns, 0, 50, "Duration of a multi-qubit gate (ns)")
        self.readout = _box(0, 0.5, ref.readout_error, 4, 0.005, "Chance that a measured bit flips")
        for title, box in (("Depolarizing, 1-qubit", self.p1), ("Depolarizing, 2-qubit", self.p2),
                           ("T1 (µs)", self.t1), ("T2 (µs)", self.t2), ("Gate time 1q (ns)", self.g1),
                           ("Gate time 2q (ns)", self.g2), ("Readout error", self.readout)):
            box.valueChanged.connect(lambda _v: self.changed.emit())
            rows.addRow(title, box)
        self.add(self.fields)
        self.note = label("", "CardCaption", wrap=True)
        self.add(self.note)
        self._mode_changed()

    def _mode_changed(self) -> None:
        self.fields.setVisible(self.noisy)
        self.note.setText("Noise follows every gate on the qubits it touches; idle qubits are not damped. "
                          "The starting values are illustrative, not a calibration of any real device."
                          if self.noisy else "The exact state vector: no errors, no decoherence.")
        self.changed.emit()

    @property
    def noisy(self) -> bool:
        return self.mode.current == "noisy"

    def model(self) -> NoiseModel:
        """The noise model the fields describe; ``ValueError`` in words when they do not fit."""
        relax = self.t1.value() > 0
        return NoiseModel(self.p1.value(), self.p2.value(), self.t1.value() if relax else None,
                          self.t2.value() if relax else None, self.g1.value(), self.g2.value(),
                          self.readout.value())

    def settings(self) -> dict:
        """The fields as they stand (also when they do not fit together yet)."""
        relax = self.t1.value() > 0
        return {"mode": self.mode.current, "depolarizing_1q": self.p1.value(),
                "depolarizing_2q": self.p2.value(),
                "t1_us": self.t1.value() if relax else None, "t2_us": self.t2.value() if relax else None,
                "gate_time_1q_ns": self.g1.value(), "gate_time_2q_ns": self.g2.value(),
                "readout_error": self.readout.value()}

    def load(self, settings: dict | None) -> None:
        settings = settings or {"mode": "ideal"}
        for key, box in (("depolarizing_1q", self.p1), ("depolarizing_2q", self.p2), ("t1_us", self.t1),
                         ("t2_us", self.t2), ("gate_time_1q_ns", self.g1), ("gate_time_2q_ns", self.g2),
                         ("readout_error", self.readout)):
            if key in settings:
                box.setValue(settings[key] or 0.0)
        self.mode.select(settings.get("mode", "ideal"))
