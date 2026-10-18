"""Wave mechanics: a Gaussian wave packet in a potential you choose, advanced in time by the
split-step Fourier method — watch it tunnel, reflect, spread or oscillate — with its norm,
position, momentum, energy and transmission; and the potential's lowest stationary states."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labs.simulation.quantum.waves import POTENTIALS, USES, WaveSetup, WaveSimulation, eigenstates

from ...icons import icon
from ...widgets import Card, DataTable, KeyValues, ResponsiveRow, label
from ...widgets.sim_views import LineChart

__all__ = ["WaveLab"]


def _spin(low: float, high: float, step: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


class WaveLab(QWidget):
    kind = "waves"
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.simulation: WaveSimulation | None = None
        self.peak = 1.0
        self.timer = QTimer(self, interval=33)
        self.timer.timeout.connect(self.tick)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        setup = Card("Set-up", "ħ = m = 1")
        form = QFormLayout()
        self.potential = QComboBox()
        for key, title in POTENTIALS.items():
            self.potential.addItem(title, key)
        self.potential.setCurrentIndex(self.potential.findData("barrier"))
        self.height_spin = _spin(-20, 20, 0.1, 1.0)
        self.width_spin = _spin(0.1, 40, 0.1, 1.0)
        self.omega = _spin(0.01, 5, 0.05, 0.5)
        self.x0 = _spin(-100, 100, 0.5, -15.0)
        self.sigma = _spin(0.2, 30, 0.1, 2.0)
        self.k0 = _spin(-10, 10, 0.05, 1.2)
        self.points = QComboBox()
        for value in (512, 1024, 2048, 4096):
            self.points.addItem(str(value), value)
        self.points.setCurrentIndex(1)
        self.dt = _spin(0.001, 0.2, 0.005, 0.02, 3)
        self.absorbing = QCheckBox("Absorb at the edges")
        self.absorbing.setChecked(True)
        for title, widget in (("Potential", self.potential), ("Height / depth V₀", self.height_spin),
                              ("Width a", self.width_spin), ("Oscillator ω", self.omega),
                              ("Packet centre x₀", self.x0), ("Packet width σ", self.sigma),
                              ("Wave number k₀", self.k0), ("Grid points", self.points),
                              ("Time step", self.dt),
                              ("", self.absorbing)):
            form.addRow(title, widget)
        setup.add(form)
        row = QHBoxLayout()
        self.play_button = QPushButton(icon("play"), "Play")
        self.play_button.setObjectName("Primary")
        self.play_button.clicked.connect(self.toggle)
        self.step_button = QPushButton(icon("step-forward"), "Step")
        self.step_button.clicked.connect(lambda: self.tick(force=True))
        self.reset_button = QPushButton(icon("skip-back"), "Reset")
        self.reset_button.clicked.connect(self.reset)
        self.speed = QSpinBox()
        self.speed.setRange(1, 200)
        self.speed.setValue(10)
        self.speed.setPrefix("steps/frame ")
        for widget in (self.play_button, self.step_button, self.reset_button, self.speed):
            row.addWidget(widget)
        setup.add(row)

        readout = Card("Now", "")
        self.values = KeyValues(())
        readout.add(self.values)
        self.energy_note = label("", "CardCaption", wrap=True)
        readout.add(self.energy_note)
        box.addWidget(ResponsiveRow([(setup, 0), (readout, 1)], breakpoint=760))

        self.density = LineChart("Probability density |ψ(x, t)|²", "x", "|ψ|²")
        self.potential_chart = LineChart("Potential V(x)", "x", "V")
        self.potential_chart.setMinimumHeight(130)
        charts = Card("The packet", "the shaded band is where the potential stands")
        charts.add(self.density)
        charts.add(self.potential_chart)
        box.addWidget(charts)

        levels = Card("Stationary states", "the lowest energies of this potential (finite differences)")
        row = QHBoxLayout()
        self.level_count = QSpinBox()
        self.level_count.setRange(1, 30)
        self.level_count.setValue(6)
        self.level_count.setPrefix("states ")
        self.levels_button = QPushButton(icon("sigma"), "Compute")
        self.levels_button.clicked.connect(self.compute_levels)
        row.addWidget(self.level_count)
        row.addWidget(self.levels_button)
        row.addStretch(1)
        levels.add(row)
        self.level_table = DataTable(("n", "Energy Eₙ", "Eₙ − Eₙ₋₁"), mono_columns=(0, 1, 2),
                                     stretch_column=1)
        self.level_table.cellClicked.connect(lambda r, _c: self.show_level(r))
        levels.add(self.level_table)
        self.level_chart = LineChart("Stationary state ψₙ(x)", "x", "ψ")
        levels.add(self.level_chart)
        box.addWidget(levels)
        box.addStretch(1)
        self._levels = None
        for widget in (self.height_spin, self.width_spin, self.omega, self.x0, self.sigma, self.k0, self.dt):
            widget.valueChanged.connect(lambda _v: self.reset())
        for combo in (self.potential, self.points):
            combo.currentIndexChanged.connect(lambda _i: self.reset())
        self.absorbing.toggled.connect(lambda _on: self.reset())
        self.potential.currentIndexChanged.connect(lambda _i: self._show_uses())
        self._show_uses()
        self.reset()

    def _show_uses(self) -> None:
        """Only the values the chosen potential depends on can be changed."""
        used = USES[self.potential.currentData()]
        for name, spin in (("height", self.height_spin), ("width", self.width_spin), ("omega", self.omega)):
            spin.setEnabled(name in used)
            spin.setToolTip("" if name in used else f"Not used by the {self.potential.currentText().lower()}")

    # Running ---------------------------------------------------------------------------------------
    def setup(self) -> WaveSetup:
        return WaveSetup(self.potential.currentData(), self.height_spin.value(), self.width_spin.value(),
                         self.omega.value(), self.x0.value(), self.sigma.value(), self.k0.value(),
                         points=int(self.points.currentData()), dt=self.dt.value(),
                         absorbing=self.absorbing.isChecked())

    def reset(self) -> None:
        try:
            self.simulation = WaveSimulation(self.setup())
        except ValueError as exc:
            self.message.emit(str(exc))
            return
        self.peak = float(self.simulation.density().max()) * 1.15
        v = self.simulation.v
        shown = np.clip(v, -50, 50)
        self.potential_chart.set_line(self.simulation.x, shown, fill=True, band=self._band(),
                                      headline=self.potential.currentText())
        self._levels = None
        self.level_table.clear_rows()
        self.level_chart.set_line(np.zeros(0), np.zeros(0))
        self.draw()

    def _band(self) -> tuple[float, float] | None:
        kind, half = self.potential.currentData(), self.width_spin.value() / 2
        if kind in ("barrier", "well", "box"):
            return (-half, half)
        if kind == "step":
            return (0.0, float(self.simulation.x[-1]) if self.simulation else 40.0)
        return None

    def toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()
        self.play_button.setText("Pause" if self.timer.isActive() else "Play")
        self.play_button.setIcon(icon("pause" if self.timer.isActive() else "play"))

    def tick(self, force: bool = False) -> None:
        if self.simulation is None:
            return
        self.simulation.step(self.speed.value())
        self.draw()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.timer.isActive():  # no animation nobody can see
            self.toggle()
        super().hideEvent(event)

    def draw(self) -> None:
        sim = self.simulation
        if sim is None:
            return
        self.density.set_line(sim.x, sim.density(), y_range=(0.0, self.peak), fill=True, band=self._band(),
                              headline=f"t = {sim.time:.2f}")
        half = self.width_spin.value() / 2
        kind = self.potential.currentData()
        edge = 0.0 if kind == "step" else half
        flow = kind in ("barrier", "step")
        rows = [("Time t", f"{sim.time:.3f}"), ("Norm ∫|ψ|²", f"{sim.norm():.5f}"),
                ("⟨x⟩", f"{sim.mean_x():+.3f}"), ("⟨p⟩", f"{sim.mean_p():+.4f}"),
                ("⟨E⟩", f"{sim.energy():.5f}"),
                ("Transmitted", f"{sim.beyond(edge):.4f}" if flow else "—"),
                ("Reflected", f"{sim.before(0.0 if kind == 'step' else -half):.4f}" if flow else "—")]
        for key, value in rows:
            self.values.set_value(key, value)
        energy = 0.5 * self.k0.value() ** 2 + 1 / (8 * self.sigma.value() ** 2)
        if kind in ("barrier", "step"):
            versus = "below" if energy < self.height_spin.value() else "above"
            ending = ": what gets through tunnels." if versus == "below" else "."
            self.energy_note.setText(f"The packet's energy ≈ {energy:.3f} is {versus} the barrier height "
                                     f"{self.height_spin.value():.3f}{ending}")
        else:
            self.energy_note.setText(f"The packet's energy ≈ {energy:.3f}.")

    # Stationary states ------------------------------------------------------------------------------
    def compute_levels(self) -> None:
        try:
            energies, x, states = eigenstates(self.setup(), self.level_count.value())
        except ValueError as exc:
            self.message.emit(str(exc))
            return
        self._levels = (energies, x, states)
        self.level_table.clear_rows()
        for n, energy in enumerate(energies):
            gap = f"{energy - energies[n - 1]:.5f}" if n else ""
            self.level_table.add_row([str(n), f"{energy:.6f}", gap])
        self.show_level(0)

    def show_level(self, n: int) -> None:
        if self._levels is None or not 0 <= n < len(self._levels[0]):
            return
        energies, x, states = self._levels
        self.level_chart.set_line(x, states[:, n], headline=f"n = {n}, E = {energies[n]:.5f}")

    # Saving ------------------------------------------------------------------------------------------
    def settings(self) -> dict:
        elapsed = self.simulation.time if self.simulation else 0.0
        return {"setup": self.setup().to_dict(), "duration": round(elapsed, 4)}

    def load(self, settings: dict) -> None:
        setup = WaveSetup.from_dict(settings.get("setup", {}))
        widgets = [(self.height_spin, setup.height), (self.width_spin, setup.width),
                   (self.omega, setup.omega), (self.x0, setup.x0), (self.sigma, setup.sigma),
                   (self.k0, setup.k0), (self.dt, setup.dt)]
        for widget, value in widgets:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        for combo, value in ((self.potential, setup.potential), (self.points, setup.points)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.absorbing.blockSignals(True)
        self.absorbing.setChecked(setup.absorbing)
        self.absorbing.blockSignals(False)
        self.reset()
        duration = float(settings.get("duration", 0))
        if self.simulation is not None and duration > 0:
            self.simulation.step(int(round(duration / setup.dt)))
            self.draw()

    def outputs(self) -> dict[str, bytes | str]:
        sim = self.simulation
        if sim is None:
            return {}
        rows = ["x,density,potential"] + [f"{x:.6f},{d:.8e},{v:.6f}" for x, d, v in
                                          zip(sim.x, sim.density(), sim.v, strict=True)]
        return {"density.csv": "\n".join(rows) + "\n"}

    def default_name(self) -> str:
        return self.potential.currentText()
