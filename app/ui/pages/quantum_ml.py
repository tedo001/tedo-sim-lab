"""Quantum ML lab: variational quantum classifiers and quantum-kernel SVMs, simulated exactly on the
CPU (``labs/quantum_ml``) and run like any experiment (queue, tracking, Training and Evaluation
pages). A finished run shows its decision boundary, its circuit with the trained angles and its
kernel matrix, and opens on Model Testing for what-if sweeps."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QScrollArea, QWidget

from core.common.vocab import Task
from labs.quantum_ml.presets import QUANTUM_PRESETS

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, KeyValues, Page, ResponsiveRow, label
from .lab_parts import LabDatasets, PresetGrid
from .quantum_views import BoundaryView, CircuitDiagram, KernelView

__all__ = ["QuantumMlPage"]


class QuantumMlPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Quantum ML", "variational circuits and quantum kernels, simulated exactly", parent)
        self.ctx = ctx
        builder = QPushButton(icon("sliders-horizontal"), "Experiment Builder")
        builder.clicked.connect(lambda: ctx.navigate("experiment_builder"))
        self.head.add_action(builder)
        intro = Card("How it works")
        intro.add(label("Each feature is scaled to an angle in 0 … π and written into a qubit (angle "
                        "encoding: RY(x); ZZ encoding: Hadamards, RZ(2x) and entangling RZ terms on every "
                        "pair). A variational classifier then applies layers of trainable RY/RZ rotations "
                        "and a ring of CNOTs, reads ⟨Z⟩ of one qubit per class and learns by the "
                        "parameter-shift rule with Adam. A quantum-kernel SVM instead compares samples by "
                        "the overlap |⟨φ(x)|φ(x′)⟩|² of their encoded states. Up to ten qubits are simulated "
                        "exactly; no quantum hardware or account is needed.", "Body",
                        wrap=True))
        self.body.addWidget(intro)
        self.presets = PresetGrid(ctx, QUANTUM_PRESETS, "each runs in seconds to a minute on a CPU")
        self.body.addWidget(self.presets)
        self.body.addWidget(self._results_card())
        self.datasets = LabDatasets(ctx, lambda card: Task.QUANTUM_CLASSIFICATION in card.tasks,
                                    "synthetic sets are made on your computer; tables need no download")
        self.body.addWidget(self.datasets)
        self.body.addStretch(1)
        ctx.experiments.run_changed.connect(lambda _id: self.refresh())
        self.refresh()

    def _results_card(self) -> Card:
        card = Card("A finished run", "boundary, circuit and kernel")
        row = QHBoxLayout()
        self.runs = QComboBox()
        self.runs.setMinimumWidth(220)
        self.runs.setMaximumWidth(420)
        self.runs.currentIndexChanged.connect(lambda _: self.show_run())
        row.addWidget(self.runs, 1)
        self.testing = QPushButton(icon("test-tube"), "Test on Model Testing")
        self.testing.clicked.connect(self.open_testing)
        row.addWidget(self.testing)
        card.add(row)
        self.empty = label("Run a preset; its results appear here when it finishes.", "CardCaption",
                           wrap=True)
        card.add(self.empty)
        self.values = KeyValues((("Model", ""), ("Test accuracy", ""), ("Qubits", ""), ("Angles", ""),
                                 ("Kernel alignment", "")))
        card.add(self.values)
        self.boundary = BoundaryView()
        self.boundary_note = label("", "CardCaption", wrap=True)
        column = Card("Decision boundary",
                      "the class predicted at each pair of angles; dots are test samples")
        column.add(self.boundary)
        column.add(self.boundary_note)
        self.kernel = KernelView()
        kernel_card = Card("Kernel matrix", "training samples sorted by class; light = similar states")
        kernel_card.add(self.kernel)
        self.kernel_card = kernel_card
        self.views = ResponsiveRow([(column, 3), (kernel_card, 2)])
        card.add(self.views)
        self.circuit = CircuitDiagram()
        scroll = QScrollArea()
        scroll.setWidget(self.circuit)
        scroll.setWidgetResizable(False)
        scroll.setMinimumHeight(80)
        scroll.setMaximumHeight(400)
        self.circuit_card = Card("Circuit", "data angles by name, trained angles in radians")
        self.circuit_card.add(scroll)
        card.add(self.circuit_card)
        return card

    def refresh(self) -> None:
        self.presets.refresh()
        current = self.runs.currentData()
        self.runs.blockSignals(True)
        self.runs.clear()
        for view in self.ctx.experiments.views():
            if view.task == Task.QUANTUM_CLASSIFICATION.value and view.status in ("completed",
                                                                                  "early_stopped") \
                    and (view.run_dir / "results.json").is_file():
                self.runs.addItem(f"{view.name} · {view.model} · {view.created_at[:16]}", view.id)
        self.runs.setCurrentIndex(max(self.runs.findData(current), 0))
        self.runs.blockSignals(False)
        self.show_run()

    def show_run(self) -> None:
        run_id = self.runs.currentData()
        has = run_id is not None
        for widget in (self.values, self.views, self.circuit_card, self.testing):
            widget.setVisible(has)
        self.empty.setVisible(not has)
        if not has:
            return
        view = self.ctx.experiments.view(run_id)
        try:
            results: dict[str, Any] = json.loads((view.run_dir / "results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        metrics = results.get("metrics", {})
        circuit = results.get("circuit")
        kernel = results.get("kernel")
        self.values.set_value("Model", "Variational classifier" if circuit else "Quantum-kernel SVM")
        self.values.set_value("Test accuracy", f"{metrics.get('test_acc', 0):.1%}")
        self.values.set_value("Qubits", str(circuit["qubits"]) if circuit else str(len(results.get("angles",
                                                                                                   []))))
        kept = results.get("kept_variance", 1.0)
        self.values.set_value("Angles", ", ".join(results.get("angles", [])) +
                              (f" ({kept:.0%} of the variance)" if kept < 1 else ""))
        self.values.set_value("Kernel alignment", f"{kernel['alignment']:.3f}" if kernel else "—")
        self.boundary.set_boundary(results.get("boundary"), results.get("classes", []))
        self.boundary_note.setText("" if results.get("boundary") else
                                   "Drawn for two-feature datasets; this one has more angles.")
        self.kernel.set_matrix(kernel["matrix"] if kernel else None)
        self.kernel_card.setVisible(bool(kernel))
        params = None
        best = view.run_dir / "checkpoints" / "best.npz"
        if circuit and best.is_file():
            with np.load(best) as saved:
                params = saved["params"] if "params" in saved.files else None
        self.circuit.set_circuit(circuit, params, list(results.get("angles", [])))
        self.circuit_card.setVisible(bool(circuit))

    def open_testing(self) -> None:
        run_id = self.runs.currentData()
        if run_id is not None:
            self.ctx.show_page("testing").open_run(run_id)
