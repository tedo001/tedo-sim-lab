"""The Experiment Builder's *Simulated device* card for quantum classifiers: an ideal simulation, or
a noisy one with the errors of an imperfect device (depolarizing errors after one- and two-qubit
gates, T1/T2 relaxation for each gate's duration, readout errors) and shots; and the classical
baselines fitted on the same split. Every setting is recorded in ``training.simulator``."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from core.experiment_engine.spec import QuantumTrainingSpec, SimulatorSpec

from ...widgets import Card, label
from .fields import combo, double, form, spin

__all__ = ["DeviceCard"]

BASELINE_TITLES = (("logistic_regression", "Logistic regression"), ("svm_rbf", "SVM (RBF)"),
                   ("mlp", "Small neural network"))


class DeviceCard(Card):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Simulated device", "never hardware: ideal or noisy simulation", parent=parent)
        rows = form(self)
        self.mode = combo([("Ideal simulation", "ideal"), ("Noisy hardware simulation", "noisy")])
        self.mode.currentIndexChanged.connect(lambda _: self._enable())
        rows.addRow("Simulation", self.mode)
        defaults = SimulatorSpec()
        self.p1 = double(0.0, 1.0, defaults.depolarizing_1q, 4, 0.001)
        self.p2 = double(0.0, 1.0, defaults.depolarizing_2q, 4, 0.005)
        self.t1 = double(0.0, 1e6, defaults.t1_us, 1, 10.0)
        self.t2 = double(0.0, 2e6, defaults.t2_us, 1, 10.0)
        self.t1.setSpecialValueText("no relaxation")
        self.g1 = double(1.0, 1e5, defaults.gate_time_1q_ns, 0, 10.0)
        self.g2 = double(1.0, 1e5, defaults.gate_time_2q_ns, 0, 50.0)
        self.readout = double(0.0, 0.5, defaults.readout_error, 4, 0.005)
        self.noise_fields = (self.p1, self.p2, self.t1, self.t2, self.g1, self.g2, self.readout)
        for title, widget, tip in (
                ("Depolarizing, 1-qubit gates", self.p1, "λ after every one-qubit gate"),
                ("Depolarizing, 2-qubit gates", self.p2, "λ after every CNOT, on both qubits"),
                ("T1 (µs)", self.t1, "Relaxation time; 0 = no relaxation"),
                ("T2 (µs)", self.t2, "Coherence time, at most 2 × T1"),
                ("Gate time 1q (ns)", self.g1, "How long a one-qubit gate relaxes for"),
                ("Gate time 2q (ns)", self.g2, "How long a two-qubit gate relaxes for"),
                ("Readout error", self.readout, "Chance that a measured bit flips")):
            widget.setToolTip(tip)
            rows.addRow(title, widget)
        self.shots = spin(0, 1_000_000, 0, special="exact")
        self.shots.setToolTip("Measurements per expectation value; exact = read the simulated state directly")
        rows.addRow("Shots", self.shots)
        self.baselines = {key: QCheckBox(title) for key, title in BASELINE_TITLES}
        holder = QWidget()
        line = QHBoxLayout(holder)
        line.setContentsMargins(0, 0, 0, 0)
        for key, box in self.baselines.items():
            box.setChecked(key in QuantumTrainingSpec().baselines)
            line.addWidget(box)
        line.addStretch(1)
        rows.addRow("Classical baselines", holder)
        self.add(label("Results are simulator results, never hardware results, and the report claims no "
                       "quantum advantage. Noisy simulation keeps density matrices (at most 8 qubits) and is "
                       "slower.", "CardCaption", wrap=True))
        self._enable()

    def _enable(self) -> None:
        noisy = self.mode.currentData() == "noisy"
        for widget in self.noise_fields:
            widget.setEnabled(noisy)

    def simulator(self) -> SimulatorSpec:
        relax = self.t1.value() > 0
        return SimulatorSpec(mode=self.mode.currentData(), depolarizing_1q=self.p1.value(),
                             depolarizing_2q=self.p2.value(), t1_us=self.t1.value() if relax else None,
                             t2_us=self.t2.value() if relax else None, gate_time_1q_ns=self.g1.value(),
                             gate_time_2q_ns=self.g2.value(), readout_error=self.readout.value(),
                             shots=self.shots.value() or None)

    def chosen_baselines(self) -> tuple[str, ...]:
        return tuple(key for key, box in self.baselines.items() if box.isChecked())

    def load(self, training: QuantumTrainingSpec) -> None:
        simulator = training.simulator
        self.mode.setCurrentIndex(max(self.mode.findData(simulator.mode), 0))
        self.p1.setValue(simulator.depolarizing_1q)
        self.p2.setValue(simulator.depolarizing_2q)
        self.t1.setValue(simulator.t1_us or 0.0)
        self.t2.setValue(simulator.t2_us or 0.0)
        self.g1.setValue(simulator.gate_time_1q_ns)
        self.g2.setValue(simulator.gate_time_2q_ns)
        self.readout.setValue(simulator.readout_error)
        self.shots.setValue(simulator.shots or 0)
        for key, box in self.baselines.items():
            box.setChecked(key in training.baselines)
        self._enable()
