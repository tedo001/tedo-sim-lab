"""The Simulation Lab page: building circuits by clicking, the wave packet, the image pipeline,
synthetic datasets handed to the Dataset Hub, the camera, and saving, reopening and exporting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.ui.pages.simulation import SimulationLabPage
from labs.simulation import records
from labs.simulation.vision.png import write_png


@pytest.fixture
def page(ctx, qtbot) -> SimulationLabPage:
    widget = SimulationLabPage(ctx)
    qtbot.addWidget(widget)
    widget.resize(1300, 900)
    widget.show()
    return widget


def test_circuits_are_built_by_clicking(page) -> None:
    lab = page.tools["circuit"]
    lab.clear()
    lab.select_gate("H")
    lab.canvas.clicked.emit(0, 0)
    lab.select_gate("CX")
    lab.canvas.clicked.emit(0, 0)          # the control …
    assert lab.canvas.pending == [0] and len(lab.circuit.ops) == 1
    lab.canvas.clicked.emit(0, 1)          # … then the target: column 0 is busy, so column 1
    assert [(op.gate, op.column) for op in lab.circuit.ops] == [("H", 0), ("CX", 1)]
    assert [row[0] for row in lab.bars.rows()] == ["|00⟩", "|11⟩"]
    assert all(abs(v) < 1e-9 for v in lab.spheres[0].vector)  # entangled: the Bloch vector shrinks to 0
    lab.select_gate("RX")
    lab.angle_spins[0].setValue(1.0)        # θ = π
    lab.canvas.clicked.emit(2, 0)
    assert lab.circuit.ops[-1].params == pytest.approx((np.pi,))
    lab.canvas.remove_requested.emit(2, 0)
    assert len(lab.circuit.ops) == 2
    lab.undo()
    assert len(lab.circuit.ops) == 3
    lab.qubits.setValue(1)  # only the CNOT used qubit 1
    assert lab.circuit.qubits == 1 and [op.gate for op in lab.circuit.ops] == ["H", "RX"]
    assert lab.description.text() == ""
    lab._preset_chosen(lab.presets.findText("Grover search"))
    assert lab.bars.rows()[0][0] == "|11⟩" and lab.bars.rows()[0][1] == pytest.approx(1.0)
    assert "marks |11⟩" in lab.description.text() and lab.table.rowCount() == 1
    assert lab.presets.currentIndex() == 0 and lab.qubits.value() == 2


def test_the_wave_packet_moves_and_levels_are_computed(page) -> None:
    page.show_tool("waves")
    lab = page.tools["waves"]
    lab.speed.setValue(50)
    start = float(lab.values.value_text("⟨x⟩"))
    for _ in range(10):
        lab.tick()
    assert float(lab.values.value_text("Time t")) == pytest.approx(10.0)
    assert float(lab.values.value_text("⟨x⟩")) > start and float(lab.values.value_text("Norm ∫|ψ|²")) > 0.99
    lab.toggle()
    assert lab.timer.isActive() and lab.play_button.text() == "Pause"
    page.show_tool("circuit")  # hidden: the animation stops
    assert not lab.timer.isActive()
    page.show_tool("waves")
    lab.potential.setCurrentIndex(lab.potential.findData("harmonic"))
    lab.compute_levels()
    energies = [float(lab.level_table.item(row, 1).text()) for row in range(lab.level_table.rowCount())]
    assert energies[:3] == pytest.approx([0.25, 0.75, 1.25], abs=2e-3)
    saved = lab.settings()
    lab.potential.setCurrentIndex(lab.potential.findData("free"))
    lab.load(saved)
    assert lab.potential.currentData() == "harmonic"


def test_the_image_pipeline_runs_in_the_background(page, qtbot, tmp_path) -> None:
    page.show_tool("image")
    lab = page.tools["image"]
    qtbot.waitUntil(lambda: lab.result is not None, timeout=10_000)
    assert lab.result.ndim == 2  # Canny edges are grey
    lab.add_step("invert")
    qtbot.waitUntil(lambda: lab.result is not None and lab.result.mean() > 0.5, timeout=10_000)
    lab.list.setCurrentRow(0)
    lab.form.editors["sigma"].setValue(4.0)
    assert lab.steps[0].params["sigma"] == 4.0
    lab.list.item(1).setCheckState(lab.list.item(1).checkState().__class__.Unchecked)
    assert lab.steps[1].enabled is False
    picture = tmp_path / "photo.png"
    write_png(picture, np.random.default_rng(0).random((40, 60, 3)))
    assert lab.open_file(picture) and lab.image.shape == (40, 60, 3)
    assert not lab.open_file(tmp_path / "missing.png")
    lab.clear_steps()
    qtbot.waitUntil(lambda: lab.result is not None and lab.result.shape == (40, 60, 3), timeout=10_000)
    lab.use_result()
    assert lab.source == {"result_of": []}


def test_synthetic_datasets_open_in_the_dataset_hub(window, qtbot) -> None:
    window.navigate("simulation_lab")
    page = window.page_widget("simulation_lab")
    page.show_tool("scenes")
    lab = page.tools["scenes"]
    lab.count.setValue(4)
    lab.name.setText("test shapes")
    lab.generate()
    qtbot.waitUntil(lambda: lab.dataset is not None, timeout=20_000)
    coco = json.loads(lab.dataset.read_text())
    assert len(coco["images"]) == 4 and lab.dataset.parent.name == "test-shapes"
    lab.generate()  # the same name again is refused, not overwritten
    assert "already holds" in page.message.text()
    lab.count.setValue(100_000)
    lab.name.setText("too many")
    lab.generate()
    assert lab.generate_button.text() == "Stop"
    lab.generate()  # the same button stops it
    qtbot.waitUntil(lambda: lab._job is None, timeout=20_000)
    assert "nothing was kept" in lab.progress.text()
    assert not list(lab._folder().parent.glob("*too-many*"))
    assert lab.generate_button.text() == "Make the dataset"
    lab.inspect()
    assert window.current_page_id() == "dataset_hub"
    hub = window.page_widget("dataset_hub")
    assert hub.switch.current == "coco"


def test_the_camera_and_saving_reopening_exporting(page, ctx, tmp_path) -> None:
    page.show_tool("camera")
    camera = page.tools["camera"]
    camera.spins["k1"].setValue(0.3)
    assert "P = K [R | t]" in camera.matrices.toPlainText() and camera.view.edges
    folder = page.save("barrel lens")
    assert folder is not None and (folder / "matrices.txt").is_file()
    page.show_tool("circuit")
    page.tools["circuit"].set_circuit(page.tools["circuit"].circuit)  # (keep the Bell preset)
    assert page.save("bell") is not None
    page.show_tool("saved")
    assert page.saved.table.rowCount() == 2
    camera.spins["k1"].setValue(0.0)
    row = next(i for i, item in enumerate(page.saved.items) if item.kind == "camera")
    page.saved.open_row(row)
    assert page.switch.current == "camera" and camera.spins["k1"].value() == pytest.approx(0.3)
    script = page.export_script(tmp_path / "camera_sim.py")
    assert script.is_file() and "projection_matrix" in script.read_text()
    page.show_tool("saved")
    page.saved.delete_row(row, confirm=False)
    assert len(records.list_saved(ctx.paths.results)) == 1
    assert not page.save_button.isEnabled()  # nothing to save on the list itself


def test_the_lattice_is_solved_drawn_in_3d_saved_and_exported(page, ctx, qtbot, tmp_path) -> None:
    lab = page.tools["lattice"]
    lab.cells.setValue(2)
    lab.spacing.setValue(4.0)
    lab.cutoff.setValue(5.0)
    assert lab.size_note.text().startswith("4 ions, 4 electrons")
    page.show_tool("lattice")  # the first showing solves in the background
    qtbot.waitUntil(lambda: lab.result is not None, timeout=60_000)
    result = lab.result
    assert result.converged and lab.status.text().startswith("Converged")
    assert lab.values.value_text("Electrons") == "4 on 4 ions"
    views = lab.views
    if views.timer.isActive():  # it plays when shown; the test moves it by hand
        views.toggle()
    assert len(views.left.points) == 4 and views.left._data is not None
    before = views.walkers.positions.copy()
    for _ in range(20):
        views.tick()
    assert not np.allclose(before, views.walkers.positions)
    views.slider.setValue(0)
    assert views.iteration_label.text() == "the first guess"
    first = lab.cut.charts["v_hartree"].ys.copy()
    views.slider.setValue(views.slider.maximum())
    assert views.iteration_label.text() == "self-consistent"
    assert not np.allclose(first, lab.cut.charts["v_hartree"].ys)  # the density moved, so V_H did
    eff = lab.cut.charts["v_eff"].ys
    parts = sum(lab.cut.charts[key].ys for key in ("v_ion", "v_hartree", "v_xc"))
    assert np.allclose(eff, parts)  # V_eff = V_H + V_ion + V_xc, point by point
    for index in range(views.show_box.count()):
        views.show_box.setCurrentIndex(index)
        assert views.right._data is not None
    lab.convergence.state_chosen.emit(1)
    assert views.show_box.currentData() == "state" and views.right.title.startswith("State 1")
    views.left.set_camera(12.0, 40.0, 1.3, notify=True)
    assert (views.right.azimuth, views.right.elevation) == (12.0, 40.0)  # the views turn together
    assert lab.convergence.levels.rowCount() >= 2
    lab.charge.setValue(2)
    assert lab.status.text().startswith("Changed")
    lab.charge.setValue(1)
    folder = page.save("square lattice")
    assert folder is not None
    for name in ("fields.csv", "levels.csv", "convergence.csv", "lattice-3d.png"):
        assert (folder / name).is_file(), name
    saved = json.loads((folder / records.FILE).read_text(encoding="utf-8"))
    assert saved["kind"] == "lattice" and saved["settings"]["setup"]["cells"] == 2
    script = page.export_script(tmp_path / "lattice_sim.py")
    assert "solve(SETUP" in script.read_text()
    lab.cells.setValue(3)
    page.open_saved("lattice", saved["settings"])  # reopening solves it again
    assert lab.cells.value() == 2
    qtbot.waitUntil(lambda: lab._job is None, timeout=60_000)
    assert lab.result is not result and lab.result.converged


def test_a_lattice_solve_can_be_stopped_and_impossible_set_ups_are_refused(page, qtbot) -> None:
    lab = page.tools["lattice"]
    lab.cells.setValue(6)
    lab.spacing.setValue(8.0)
    assert lab.size_note.text().startswith("Not possible") and not lab.solve_button.isEnabled()
    lab.spacing.setValue(4.0)
    lab.cutoff.setValue(4.0)
    lab.solve()
    assert lab.solve_button.text() == "Stop"
    lab.solve()  # the same button stops it
    qtbot.waitUntil(lambda: lab._job is None, timeout=60_000)
    assert lab.solve_button.text() == "Solve"
