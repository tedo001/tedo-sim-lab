"""Electrons in a crystal lattice: a self-consistent Kohn–Sham calculation in two dimensions,
solved in the background and drawn in 3-D — the lattice potential with the electrons moving on
it, and the averaged electron density they make — with V_eff = V_H + V_ion + V_xc taken apart."""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Signal
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

from labs.simulation.quantum.kohn_sham import ENERGY_TERMS, KohnShamResult, solve
from labs.simulation.quantum.lattice import FUNCTIONALS, LATTICES, LatticeSetup

from ....services.jobs import JobQueue
from ...icons import icon
from ...widgets import Card, KeyValues, ResponsiveRow, label
from .lattice_views import SHOW, Convergence, ElectronViews, PotentialCut

__all__ = ["LatticeLab"]


def _spin(low: float, high: float, step: float, value: float, decimals: int = 2,
          suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def _whole(low: int, high: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    return spin


class LatticeLab(QWidget):
    kind = "lattice"
    message = Signal(str)

    def __init__(self, jobs: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.result: KohnShamResult | None = None
        self._job: str | None = None
        self._asked = False  # solved once on first showing
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        defaults = LatticeSetup()

        setup = Card("Set-up", "atomic units: bohr, hartree")
        form = QFormLayout()
        self.lattice = QComboBox()
        for key, title in LATTICES.items():
            self.lattice.addItem(title, key)
        self.lattice.setMinimumWidth(150)
        self.cells = _whole(1, 6, defaults.cells)
        self.spacing = _spin(2, 12, 0.25, defaults.spacing, suffix=" bohr")
        self.charge = _whole(1, 4, defaults.charge)
        self.height_spin = _spin(0.2, 5, 0.05, defaults.height, suffix=" bohr")
        self.vacancy = QCheckBox("Leave out the middle ion (a vacancy)")
        self.functional = QComboBox()
        for key, title in FUNCTIONALS.items():
            self.functional.addItem(title, key)
        self.functional.setMinimumWidth(150)
        self.cutoff = _spin(2, 40, 1, defaults.cutoff, 1, " Ha")
        self.smearing = _spin(0.001, 0.2, 0.005, defaults.smearing, 3, " Ha")
        self.mixing = _spin(0.05, 1, 0.05, defaults.mixing)
        self.iterations = _whole(1, 500, defaults.max_iterations)
        rows = (("Lattice", self.lattice), ("Cells along x", self.cells),
                ("Lattice constant a", self.spacing), ("Ion charge Z", self.charge),
                ("Ion height d", self.height_spin), ("", self.vacancy),
                ("Exchange-correlation", self.functional), ("Plane-wave cutoff", self.cutoff),
                ("Smearing kT", self.smearing), ("Mixing", self.mixing), ("Most iterations", self.iterations))
        for title, widget in rows:
            form.addRow(title, widget)
        setup.add(form)
        self.size_note = label("", "CardCaption", wrap=True)
        setup.add(self.size_note)
        actions = QHBoxLayout()
        self.solve_button = QPushButton(icon("atom"), "Solve")
        self.solve_button.setObjectName("Primary")
        self.solve_button.setToolTip("Iterate the Kohn–Sham equations to self-consistency "
                                     "(in the background)")
        self.solve_button.clicked.connect(self.solve)
        actions.addWidget(self.solve_button)
        actions.addStretch(1)
        setup.add(actions)
        self.status = label("", "CardCaption", wrap=True)
        setup.add(self.status)

        readout = Card("Ground state", "")
        self.values = KeyValues(())
        readout.add(self.values)
        self.verdict = label("Solve to see the ground state.", "CardCaption", wrap=True)
        readout.add(self.verdict)
        box.addWidget(ResponsiveRow([(setup, 0), (readout, 1)], breakpoint=760))

        self.views = ElectronViews()
        self.cut = PotentialCut()
        self.convergence = Convergence()
        self.views.iteration_changed.connect(self._iteration)
        self.convergence.state_chosen.connect(self.views.show_state)
        for widget in (self.views, self.cut, self.convergence):
            box.addWidget(widget)
        box.addStretch(1)
        for spin in (self.cells, self.charge, self.iterations):
            spin.valueChanged.connect(lambda _v: self._changed())
        for spin in (self.spacing, self.height_spin, self.cutoff, self.smearing, self.mixing):
            spin.valueChanged.connect(lambda _v: self._changed())
        for combo in (self.lattice, self.functional):
            combo.currentIndexChanged.connect(lambda _i: self._changed())
        self.vacancy.toggled.connect(lambda _on: self._changed())
        jobs.job_finished.connect(self._solved)
        jobs.job_progress.connect(self._progress)
        self._changed(initial=True)

    # Set-up ------------------------------------------------------------------------------------------------
    def setup(self) -> LatticeSetup:
        return LatticeSetup(self.lattice.currentData(), self.cells.value(), self.spacing.value(),
                            self.charge.value(), self.height_spin.value(), self.functional.currentData(),
                            self.vacancy.isChecked(), self.cutoff.value(), self.smearing.value(),
                            self.mixing.value(), self.iterations.value())

    def _changed(self, initial: bool = False) -> None:
        setup = self.setup()
        try:
            setup.check()
        except ValueError as exc:
            self.size_note.setText(f"Not possible: {exc}.")
            self.solve_button.setEnabled(self._job is not None)
            return
        ions, electrons, waves = setup.counts()
        self.size_note.setText(f"{ions} ions, {electrons} electrons, {waves} plane waves")
        self.solve_button.setEnabled(True)
        if self.result is not None and not initial and self._job is None:
            self.status.setText("Changed: Solve again to see this set-up.")

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        if not self._asked and self.result is None and self._job is None:
            self._asked = True
            self.solve()

    # Solving ---------------------------------------------------------------------------------------------
    def solve(self) -> None:
        """Solve in the background; while it runs, the same button stops it."""
        if self._job is not None:
            self.jobs.cancel(self._job)
            return
        setup = self.setup()
        try:
            setup.check()
        except ValueError as exc:
            self.message.emit(f"Not solved: {exc}")
            return
        self.status.setText("Solving…")
        self._job = self.jobs.submit_task(lambda cancel, progress: solve(setup, progress, cancel),
                                          title="Solve the Kohn–Sham equations", quiet=True)
        self.solve_button.setText("Stop")
        self.solve_button.setIcon(icon("square"))

    def _progress(self, job_id: str, _fraction: float, text: str) -> None:
        if job_id == self._job:
            self.status.setText(text)

    def _solved(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        self.solve_button.setText("Solve")
        self.solve_button.setIcon(icon("atom"))
        self._changed(initial=True)
        job = self.jobs.job(job_id)
        if status != "completed":
            self.status.setText("Stopped." if status == "cancelled" else f"Failed: {job.error or status}")
            return
        self.show_result(job.result)

    def show_result(self, result: KohnShamResult) -> None:
        self.result = result
        steps = len(result.iterations)
        self.status.setText(f"{'Converged' if result.converged else 'Stopped unconverged'} after {steps} "
                            f"iteration{'s' if steps != 1 else ''} in {result.seconds:.2f} s")
        energies = result.energies
        gap = result.gap()
        rows = [("Electrons", f"{result.electrons:.0f} on {len(result.ions)} ions"),
                ("Total energy", f"{energies['total']:.6f} Ha"),
                ("Per electron", f"{energies['total'] / result.electrons:.6f} Ha"),
                ("Fermi level μ", f"{result.fermi:.5f} Ha"),
                ("Gap at μ", "none: a level is partly filled (a metal)" if gap is None else f"{gap:.5f} Ha")]
        rows += [(ENERGY_TERMS[key], f"{energies[key]:.6f} Ha") for key in ENERGY_TERMS if key != "total"]
        rows += [("Basis", f"{result.plane_waves} plane waves, grid {result.density.shape[1]} × "
                           f"{result.density.shape[0]}")]
        for key, value in rows:
            self.values.set_value(key, value)
        self.verdict.setText("Self-consistent: the density makes the potential it was computed in."
                             if result.converged else
                             "Not converged: raise the iterations or lower the mixing.")
        self.views.show_result(result)
        self.convergence.show_result(result)
        if self.isVisible() and not self.views.timer.isActive():
            self.views.toggle()

    def _iteration(self, _index: int) -> None:
        if self.result is not None:
            self.cut.show_fields(self.result, self.views.fields())

    # Saving ----------------------------------------------------------------------------------------------
    def settings(self) -> dict:
        return {"setup": self.setup().to_dict(), "show": self.views.show_box.currentData()}

    def load(self, settings: dict) -> None:
        setup = LatticeSetup.from_dict(settings.get("setup", {}))
        setup.check()
        widgets = ((self.cells, setup.cells), (self.spacing, setup.spacing), (self.charge, setup.charge),
                   (self.height_spin, setup.height), (self.cutoff, setup.cutoff),
                   (self.smearing, setup.smearing), (self.mixing, setup.mixing),
                   (self.iterations, setup.max_iterations))
        for widget, value in widgets:
            widget.setValue(value)
        self.lattice.setCurrentIndex(self.lattice.findData(setup.lattice))
        self.functional.setCurrentIndex(self.functional.findData(setup.functional))
        self.vacancy.setChecked(setup.vacancy)
        show = settings.get("show", "density")
        shown = self.views.show_box.findData(show if show in SHOW else "density")
        self.views.show_box.setCurrentIndex(max(0, shown))
        self._asked = True
        if self._job is None:
            self.solve()

    def outputs(self) -> dict[str, bytes | str]:
        result = self.result
        if result is None:
            return {}
        rows = ["x,y,density,v_ion,v_hartree,v_xc,v_eff"]
        v_eff = result.v_eff
        for j, y in enumerate(result.y):
            for i, x in enumerate(result.x):
                rows.append(f"{x:.5f},{y:.5f},{result.density[j, i]:.8e},{result.v_ion[j, i]:.8f},"
                            f"{result.v_hartree[j, i]:.8f},{result.v_xc[j, i]:.8f},{v_eff[j, i]:.8f}")
        filled = zip(result.eigenvalues, result.occupations, strict=True)
        levels = ["n,energy,electrons"] + [f"{n},{e:.8f},{f:.6f}" for n, (e, f) in enumerate(filled)]
        steps = ["iteration,change,energy,fermi"] + [
            f"{s.number + 1},{s.residual:.6e},{s.energy:.8f},{s.fermi:.6f}" for s in result.iterations]
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.views.grab().save(buffer, "PNG")
        buffer.close()
        return {"fields.csv": "\n".join(rows) + "\n", "levels.csv": "\n".join(levels) + "\n",
                "convergence.csv": "\n".join(steps) + "\n", "lattice-3d.png": bytes(data.data())}

    def default_name(self) -> str:
        setup = self.setup()
        return f"{LATTICES[setup.lattice].split(' (')[0]} lattice, Z = {setup.charge}"
