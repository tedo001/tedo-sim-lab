"""Headless smoke test: open every page in :data:`app.navigation.NAV`, run each engine the
window runs itself once, then exit.

    QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test [--screenshots DIR]

Fails (exit code 1) if any page raises while being built or shown, including
errors raised later inside slots, or if an engine fails (in the installed app: a module the
build left out, found before someone clicks). ``--screenshots`` saves one PNG per page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .navigation import NAV

log = logging.getLogger("tedo.smoke")

__all__ = ["SmokeReport", "run_smoke"]


@dataclass
class SmokeReport:
    visited: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.visited) == len(NAV)

    def summary(self) -> str:
        if self.ok:
            return f"SMOKE OK - {len(self.visited)} pages: {', '.join(self.visited)}"
        return (f"SMOKE FAILED - {len(self.visited)}/{len(NAV)} pages, "
                f"{len(self.errors)} error(s):\n" + "\n".join(self.errors))


def run_smoke(window: MainWindow, app: QApplication, *, uncaught: list[str],
              screenshot_dir: Path | None = None) -> SmokeReport:
    """Visit every page; ``uncaught`` is the list the excepthook appends to."""
    report = SmokeReport()
    # A card, plugin or runner that did not load is a broken build (e.g. a module missing from
    # the installed app), even though every page still opens around it.
    report.errors += [f"catalogue: {problem}" for problem in window.ctx.catalog.errors]
    window.show()
    app.processEvents()
    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    for spec in NAV:
        try:
            window.navigate(spec.id)
            app.processEvents()
            if window.current_page_id() != spec.id:
                raise RuntimeError(f"navigate({spec.id!r}) left the window on "
                                   f"{window.current_page_id()!r}")
        except Exception as exc:  # report every page, not just the first failure
            report.errors.append(f"{spec.id}: {type(exc).__name__}: {exc}")
            log.error("Page %s failed to open", spec.id, exc_info=exc)  # the traceback, in app.log
            continue
        report.visited.append(spec.id)
        if screenshot_dir is not None:
            file = screenshot_dir / f"{len(report.visited):02d}-{spec.id}.png"
            window.grab().save(str(file))
            report.screenshots.append(file)
    try:
        _exercise_layout(window, app, screenshot_dir, report)
    except Exception as exc:
        report.errors.append(f"layout: {type(exc).__name__}: {exc}")
    try:
        exercise_engines()
    except Exception as exc:
        report.errors.append(f"engines: {type(exc).__name__}: {exc}")
        log.error("An engine failed in the smoke test", exc_info=exc)
    report.errors.extend(uncaught)
    window.close()
    return report


def _exercise_layout(window: MainWindow, app: QApplication, screenshot_dir: Path | None,
                     report: SmokeReport) -> None:
    """Collapse the sidebar and open split view, then put everything back."""
    window.navigate("home")
    window.set_sidebar_collapsed(True)
    window.set_split_open(True, "documentation")
    app.processEvents()
    if not (window.sidebar_collapsed() and window.split_open()):
        raise RuntimeError("sidebar rail or split view did not open")
    if screenshot_dir is not None:
        file = screenshot_dir / "zz-layout-rail-and-split.png"
        window.grab().save(str(file))
        report.screenshots.append(file)
    window.set_split_open(False)
    window.set_sidebar_collapsed(False)
    app.processEvents()


def exercise_engines() -> None:
    """Run, small, what the window computes in its own process: the Simulation Lab's quantum,
    wave, lattice and vision engines (numpy and scipy: every image operation, the stationary-state
    and Kohn–Sham solvers), principal components and IDX files for the 3-D explorer, the PNG
    codec, reading logs and pdb's stops."""
    import tempfile

    import numpy as np

    from core.common.log_reader import parse_lines
    from core.terminal import Parser, Screen
    from core.terminal.debugger import find_stop
    from labs.common.explore import principal_components, read_idx
    from labs.simulation.quantum import preset, sample, simulate
    from labs.simulation.quantum.density_walk import DensityWalkers
    from labs.simulation.quantum.kohn_sham import solve
    from labs.simulation.quantum.lattice import LatticeSetup
    from labs.simulation.quantum.waves import WaveSetup, WaveSimulation, eigenstates
    from labs.simulation.vision import (
        OPERATIONS,
        Camera,
        SceneSpec,
        Step,
        render_scene,
        run_pipeline,
        wireframe,
    )
    from labs.simulation.vision import sample as test_pattern
    from labs.simulation.vision.png import read_png, write_png

    state = simulate(preset("Teleportation")).state
    if abs(float(np.vdot(state, state).real) - 1.0) > 1e-9 or not sample(state, 16, 0):
        raise RuntimeError("the circuit simulator lost its norm")
    waves = WaveSimulation(WaveSetup(points=256))
    waves.step(5)
    energies, _x, _states = eigenstates(WaveSetup(potential="harmonic", points=256), 2)
    if not 0 < energies[0] < energies[1]:
        raise RuntimeError("stationary states are not ordered")
    image = test_pattern("Shapes", 48)
    result, _timings = run_pipeline(image, [Step(name) for name in sorted(OPERATIONS)])
    scene, _objects = render_scene(SceneSpec(width=48, height=48, max_size=20, seed=1),
                                   np.random.default_rng(1))
    with tempfile.TemporaryDirectory() as folder:
        if read_png(write_png(Path(folder) / "scene.png", scene)).shape != (48, 48, 3):
            raise RuntimeError("PNG round trip changed the image")
    if not wireframe(Camera()) or result.size == 0:
        raise RuntimeError("the vision engines returned nothing")
    lattice = solve(LatticeSetup(cells=2, spacing=4.0, cutoff=4.0))
    if not lattice.converged or abs(lattice.electrons - 4) > 1e-6:
        raise RuntimeError("the Kohn–Sham solver did not converge")
    DensityWalkers(lattice.density, lattice.box, lattice.ions, 4).advance(3)
    _scores, share = principal_components(np.random.default_rng(0).normal(size=(20, 4)))
    with tempfile.TemporaryDirectory() as folder:
        idx = Path(folder) / "t10k-labels-idx1-ubyte"
        idx.write_bytes(b"\x00\x00\x08\x01" + (3).to_bytes(4, "big") + bytes([7, 2, 1]))
        if read_idx(idx).tolist() != [7, 2, 1] or abs(float(share.sum()) - 1.0) > 1e-9:
            raise RuntimeError("the 3-D explorer's readers failed")
    if parse_lines(["2026-01-01 00:00:00,000 ERROR   tedo.smoke: x"])[0].level != "ERROR":
        raise RuntimeError("log lines are not read")
    screen = Screen(60, 6, 10)
    Parser(screen).feed("> /lab/x.py(3)<module>()\r\n(Pdb) ")
    if find_stop(screen) is None:
        raise RuntimeError("pdb's stop is not found")
