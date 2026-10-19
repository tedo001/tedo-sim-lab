"""The Lattice tool's views of a Kohn–Sham result: the electrons on the lattice potential and the
density they average to, in 3-D (the two views turn together); the potential's three parts along
a row of ions; and how the calculation settled, with its levels."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPushButton, QSlider, QSpinBox, QWidget

from labs.simulation.quantum.density_walk import DensityWalkers
from labs.simulation.quantum.kohn_sham import KohnShamResult

from ...icons import icon
from ...widgets import Card, DataTable, ResponsiveRow, label
from ...widgets.sim_views import LineChart
from ...widgets.surface3d import SurfaceView

__all__ = ["SHOW", "Convergence", "ElectronViews", "PotentialCut"]

DENSITY = "n (bohr⁻²)"
#: What the right-hand view shows.
SHOW = {"density": "Averaged density n(r)", "density_on_veff": "n(r) painted on V_eff",
        "v_eff": "V_eff = V_ion + V_H + V_xc", "v_hartree": "Hartree potential V_H",
        "v_xc": "Exchange-correlation V_xc", "state": "One Kohn–Sham state |ψₙ|²"}
FADE = 40  # frames an arrow stays


class ElectronViews(Card):
    """Left: V_ion with the electrons (samples of the density) moving on it, arrows where one hops
    to another ion. Right: the averaged density, a potential, or one state. The iteration slider
    goes back through the self-consistent field."""

    iteration_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Electrons in the lattice", "drag either view to turn both", parent=parent)
        self.result: KohnShamResult | None = None
        self.walkers: DensityWalkers | None = None
        self.arcs: list[list] = []
        self.state = 0
        self.timer = QTimer(self, interval=33)
        self.timer.timeout.connect(self.tick)
        controls = QHBoxLayout()
        self.play_button = QPushButton(icon("play"), "Play")
        self.play_button.setObjectName("Primary")
        self.play_button.clicked.connect(self.toggle)
        self.speed = QSpinBox()
        self.speed.setRange(1, 20)
        self.speed.setPrefix("steps/frame ")
        self.mesh = QCheckBox("Mesh")
        self.mesh.setChecked(True)
        self.mesh.toggled.connect(self._mesh)
        for widget in (self.play_button, self.speed, self.mesh):
            controls.addWidget(widget)
        controls.addStretch(1)
        self.add(controls)
        choices = QHBoxLayout()
        self.show_box = QComboBox()
        for key, title in SHOW.items():
            self.show_box.addItem(title, key)
        self.show_box.setMinimumWidth(150)
        self.show_box.currentIndexChanged.connect(lambda _i: self.draw_right())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setMinimumWidth(90)
        self.slider.valueChanged.connect(self.set_iteration)
        self.iteration_label = label("", "CardCaption")
        choices.addWidget(label("Right", "CardCaption"))
        choices.addWidget(self.show_box, 1)
        choices.addWidget(self.slider, 1)
        choices.addWidget(self.iteration_label)
        self.add(choices)
        self.left = SurfaceView("Lattice potential V_ion(r) and the electrons")
        self.right = SurfaceView(SHOW["density"])
        self.left.camera_changed.connect(lambda *view: self.right.set_camera(*view))
        self.right.camera_changed.connect(lambda *view: self.left.set_camera(*view))
        self.add(ResponsiveRow([(self.left, 1), (self.right, 1)], breakpoint=720))
        self.add(label("Each sphere is a sample of the density, not a classical electron: where the "
                       "samples spend their time is n(r). An arrow marks a hop to another ion.",
                       "CardCaption", wrap=True))

    # Data -----------------------------------------------------------------------------------------------
    def show_result(self, result: KohnShamResult) -> None:
        self.result, self.state = result, 0
        self.left.set_surface(result.x, result.y, result.v_ion, labels=("V_ion (Ha)", ""))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(result.densities) - 1)
        self.slider.setValue(len(result.densities) - 1)
        self.slider.blockSignals(False)
        self.set_iteration(self.slider.value())

    def density(self) -> np.ndarray:
        return self.result.densities[self.slider.value()]

    def set_iteration(self, index: int) -> None:
        result = self.result
        if result is None:
            return
        last = len(result.densities) - 1
        self.iteration_label.setText("self-consistent" if index == last and result.converged else
                                     f"iteration {index} of {last}" if index else "the first guess")
        count = int(min(round(result.electrons), 48))
        self.walkers = DensityWalkers(self.density(), result.box, result.ions, count, step=0.03, seed=index)
        self.arcs = []
        self.draw_left()
        self.draw_right()
        self.iteration_changed.emit(index)

    def show_state(self, index: int) -> None:
        if self.result is None or not 0 <= index < len(self.result.eigenvalues):
            return
        self.state = index
        self.show_box.setCurrentIndex(self.show_box.findData("state"))
        self.draw_right()

    def fields(self) -> dict[str, np.ndarray]:
        """The density and potentials of the iteration shown."""
        result, density = self.result, self.density()
        v_hartree, v_xc = result.potentials_of(density)
        return {"density": density, "v_hartree": v_hartree, "v_xc": v_xc,
                "v_eff": result.v_ion + v_hartree + v_xc}

    # Drawing ----------------------------------------------------------------------------------------------
    def draw_left(self) -> None:
        if self.walkers is None:
            return
        arcs = [(start, end, life / FADE) for start, end, life in self.arcs]
        self.left.set_points(self.walkers.positions, arcs)

    def draw_right(self) -> None:
        result = self.result
        if result is None:
            return
        key = self.show_box.currentData()
        fields = self.fields()
        x, y, view = result.x, result.y, self.right
        view.title = SHOW[key]
        if key == "density":
            view.set_surface(x, y, fields["density"], colour=fields["density"], scale="density",
                             labels=(DENSITY, DENSITY))
        elif key == "density_on_veff":
            view.set_surface(x, y, fields["v_eff"], colour=fields["density"], scale="density",
                             labels=("V_eff (Ha)", DENSITY))
        elif key == "state":
            state = result.state_densities[self.state]
            view.title = (f"State {self.state}: ε = {result.eigenvalues[self.state]:.4f} Ha, "
                          f"{result.occupations[self.state]:.2f} electrons")
            view.set_surface(x, y, state, colour=state, scale="density",
                             labels=("|ψ|² (bohr⁻²)", "|ψ|² (bohr⁻²)"))
        else:
            name = {"v_eff": "V_eff", "v_hartree": "V_H", "v_xc": "V_xc"}[key]
            view.set_surface(x, y, fields[key], labels=(f"{name} (Ha)", ""))

    def _mesh(self, on: bool) -> None:
        self.left.set_mesh(on)
        self.right.set_mesh(on)

    # Moving ----------------------------------------------------------------------------------------------
    def toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        elif self.walkers is not None:
            self.timer.start()
        playing = self.timer.isActive()
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.setIcon(icon("pause" if playing else "play"))

    def tick(self) -> None:
        if self.walkers is None or self.result is None:
            return
        ions, box = self.result.ions, np.array(self.result.box)
        for hop in self.walkers.advance(self.speed.value()):
            start, end = ions[hop.source], ions[hop.target]
            step = end - start
            if np.allclose(step, step - np.round(step / box) * box):  # not across the edge
                self.arcs.append([tuple(start), tuple(end), FADE])
        self.arcs = [[start, end, life - 1] for start, end, life in self.arcs[-12:] if life > 1]
        self.draw_left()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.timer.isActive():  # no animation nobody can see
            self.toggle()
        super().hideEvent(event)


class PotentialCut(Card):
    """The effective potential's three parts and their sum along a row of ions."""

    TITLES = {"v_ion": "V_ion: the ions pull", "v_hartree": "V_H: the electrons repel",
              "v_xc": "V_xc: exchange-correlation", "v_eff": "V_eff: what each electron feels"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("V_eff = V_H + V_ion + V_xc", "along a row of ions", parent=parent)
        self.charts = {key: LineChart(title, "x (bohr)", "V (Ha)") for key, title in self.TITLES.items()}
        for first, second in (("v_ion", "v_hartree"), ("v_xc", "v_eff")):
            self.add(ResponsiveRow([(self.charts[first], 1), (self.charts[second], 1)], breakpoint=620))

    def show_fields(self, result: KohnShamResult, fields: dict[str, np.ndarray]) -> None:
        self.caption.set_full_text(f"along the row of ions at y = {result.ion_y():.2f} bohr; "
                                   "at every point the sum of the first three is the fourth")
        values = {"v_ion": result.v_ion, **{key: fields[key] for key in ("v_hartree", "v_xc", "v_eff")}}
        for key, chart in self.charts.items():
            line = result.cut(values[key])
            chart.set_line(result.x, line, headline=f"{line.min():+.3f} … {line.max():+.3f}")


class Convergence(Card):
    """The density's change per iteration (log scale) and the Kohn–Sham levels; a level clicked
    is drawn in the right-hand view."""

    state_chosen = Signal(int)
    SHOWN_EMPTY = 6  # empty levels listed above the filled ones

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Self-consistency", "how far the density moved in each iteration", parent=parent)
        self.chart = LineChart("Density change per iteration", "iteration", "log₁₀ ∫|Δn| / N")
        self.levels = DataTable(("n", "Energy εₙ (Ha)", "Electrons"), mono_columns=(0, 1, 2),
                                stretch_column=1)
        self.levels.cellClicked.connect(lambda row, _column: self.state_chosen.emit(row))
        self.add(ResponsiveRow([(self.chart, 1), (self.levels, 1)], breakpoint=700))

    def show_result(self, result: KohnShamResult) -> None:
        numbers = np.array([step.number + 1 for step in result.iterations], dtype=float)
        moved = np.log10(np.maximum([step.residual for step in result.iterations], 1e-16))
        last = result.iterations[-1]
        verdict = ("converged" if result.converged else
                   "not converged: raise the iterations or lower the mixing")
        self.chart.set_line(numbers, moved, headline=f"{len(numbers)} iterations, {verdict}")
        self.caption.set_full_text(f"final change {last.residual:.1e} · "
                                   f"Fermi level μ = {result.fermi:.4f} Ha")
        self.levels.clear_rows()
        empty = 0
        for n, (energy, filled) in enumerate(zip(result.eigenvalues, result.occupations, strict=True)):
            empty += filled < 0.01
            if empty > self.SHOWN_EMPTY:
                break
            self.levels.add_row([str(n), f"{energy:+.5f}", f"{filled:.3f}"])
