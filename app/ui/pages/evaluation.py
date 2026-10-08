"""Evaluation: a finished run's scores, confusion matrix and per-class report, and
re-scoring a checkpoint on the test or validation split. scikit-learn runs also show their
curves, importance and points (:class:`SklearnResults`)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPushButton, QWidget

from labs.computer_vision.training.metrics import per_class, summary

from ...services.context import AppContext
from ..widgets import Card, ConfusionMatrixView, DataTable, Page, Pill, StatStrip, StatTile, label
from .sklearn_results import SklearnResults, read_results

__all__ = ["EvaluationPage"]


def _combo(width: int = 420) -> QComboBox:
    combo = QComboBox()
    combo.setMaximumWidth(width)
    return combo


class EvaluationPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Evaluation", "scores, confusion matrix and per-class report", parent)
        self.ctx = ctx
        self.report: dict | None = None
        self._job: str | None = None

        pick = Card("Run and evaluation")
        row = QHBoxLayout()
        self.run_combo = _combo()
        self.run_combo.currentIndexChanged.connect(lambda _: self._run_chosen())
        self.source_combo = _combo()
        self.source_combo.currentIndexChanged.connect(lambda _: self._show_source())
        row.addWidget(label("Run", "Body"))
        row.addWidget(self.run_combo, 1)
        pick.add(row)
        row = QHBoxLayout()
        self.source_label = label("Evaluation", "Body")
        row.addWidget(self.source_label)
        row.addWidget(self.source_combo, 1)
        pick.add(row)
        again = QHBoxLayout()
        self.checkpoint_combo = _combo(160)
        self.split_combo = _combo(160)
        self.evaluate_button = QPushButton("Evaluate again")
        self.evaluate_button.clicked.connect(self.evaluate)
        self.state = Pill("", "planned")
        self.state.hide()
        self.again_widgets = (label("Score the", "Body"), self.checkpoint_combo, label("on the", "Body"),
                              self.split_combo, self.evaluate_button)
        for widget in (*self.again_widgets, self.state):
            again.addWidget(widget)
        again.addStretch(1)
        pick.add(again)
        self.no_rescore = label("", "CardCaption", wrap=True)
        pick.add(self.no_rescore)
        self.empty = label("No finished runs yet. Train one from the Experiment Builder; its test results "
                           "appear here.", "CardCaption", wrap=True)
        pick.add(self.empty)
        self.body.addWidget(pick)

        self.tiles = {key: StatTile(title) for key, title in (("acc", "Accuracy"), ("precision", "Precision"),
                                                               ("recall", "Recall"), ("f1", "F1"))}
        self.stats = StatStrip(self.tiles)
        self.body.addWidget(self.stats)

        matrix_card = Card("Confusion matrix")
        self.normalise = QCheckBox("Shares of each true class")
        self.normalise.toggled.connect(lambda on: self.matrix.set_normalised(on))
        matrix_card.add_head_widget(self.normalise)
        self.matrix = ConfusionMatrixView()
        matrix_card.add(self.matrix)
        self.body.addWidget(matrix_card)

        classes = Card("Per class", padded=False)
        self.table = DataTable(("Class", "Precision", "Recall", "F1", "Samples"), mono_columns=(1, 2, 3, 4),
                               stretch_column=0)
        classes.add(self.table)
        self.body.addWidget(classes)
        self.tabular = SklearnResults()
        self.body.addWidget(self.tabular)
        self.body.addStretch(1)

        ctx.experiments.run_changed.connect(lambda _: self.refresh(keep=True))
        ctx.jobs.job_finished.connect(self._job_finished)
        self.refresh()

    # Choosing -------------------------------------------------------------------
    def refresh(self, *, keep: bool = False) -> None:
        current = self.run_combo.currentData() if keep else None
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for view in self.ctx.experiments.views(limit=200):
            if view.finished_ok and (self.ctx.experiments.evaluations(view.id) or read_results(view.run_dir)):
                self.run_combo.addItem(f"{view.name} · {view.dataset} · {view.model} · run {view.id[:8]}",
                                       view.id)
        self.run_combo.setCurrentIndex(max(self.run_combo.findData(current), 0) if current else 0)
        self.run_combo.blockSignals(False)
        has_runs = self.run_combo.count() > 0
        self.empty.setVisible(not has_runs)
        for widget in (self.run_combo, self.source_combo, self.evaluate_button, self.checkpoint_combo,
                       self.split_combo):
            widget.setEnabled(has_runs)
        self._run_chosen()

    @property
    def run_id(self) -> str | None:
        return self.run_combo.currentData()

    def _run_chosen(self, select: Path | None = None) -> None:
        view = self.ctx.experiments.view(self.run_id) if self.run_id is not None else None
        tabular = view is not None and view.tabular
        found = read_results(view.run_dir) if tabular else None
        self.tabular.setVisible(found is not None)
        if found is not None:
            self.tabular.show_results(found)
        self._fill_choices(tabular)
        task = (found or {}).get("task")
        rescorable = not tabular or task == "tabular_classification"
        self.evaluate_button.setEnabled(rescorable and view is not None and self._job is None)
        for widget in (*self.again_widgets, self.source_label, self.source_combo):
            widget.setVisible(rescorable)
        if task == "tabular_regression":
            why = "This regression run was scored once on its held-out test rows; its measures are below."
        else:
            why = ("Clustering and projections are fitted on every row, so nothing is held out to score "
                   "again; the run's measures are below.")
        self.no_rescore.setText("" if rescorable else why)
        self.no_rescore.setVisible(not rescorable)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        if self.run_id is not None:
            for title, path in self.ctx.experiments.evaluations(self.run_id):
                self.source_combo.addItem(title, str(path))
        if select is not None:
            self.source_combo.setCurrentIndex(max(self.source_combo.findData(str(select)), 0))
        self.source_combo.blockSignals(False)
        self._show_source()

    def _fill_choices(self, tabular: bool) -> None:
        """PyTorch runs re-score best or last on test or validation; scikit-learn runs have one
        fitted model and no validation split."""
        checkpoints = (("fitted model", "model"),) if tabular else (("best checkpoint", "best"),
                                                                     ("last checkpoint", "last"))
        splits = (("test split", "test"),) + (() if tabular else (("validation split", "val"),))
        for combo, items in ((self.checkpoint_combo, checkpoints), (self.split_combo, splits)):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for title, key in items:
                combo.addItem(title, key)
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.blockSignals(False)

    def _show_source(self) -> None:
        path = self.source_combo.currentData()
        self.report = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
        visible = self.report is not None
        for widget in (self.stats, self.matrix.parentWidget(), self.table.parentWidget()):
            widget.setVisible(visible)
        if not visible:
            return
        matrix, classes = self.report["matrix"], self.report["classes"]
        scores = summary(matrix)
        for key, tile in self.tiles.items():
            note = f"{int(sum(map(sum, matrix))):,} samples" if key == "acc" else "macro average"
            tile.set(f"{scores[key]:.2%}", note)
        self.matrix.set_matrix(matrix, classes)
        self.table.clear_rows()
        for name, row in zip(classes, per_class(matrix), strict=True):
            self.table.add_row((name, f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1']:.3f}",
                                f"{int(row['support']):,}"))

    # Evaluating again ---------------------------------------------------------------
    def evaluate(self) -> str | None:
        if self.run_id is None:
            return None
        try:
            self._job = self.ctx.experiments.evaluate(self.run_id, self.checkpoint_combo.currentData(),
                                                      self.split_combo.currentData())
        except FileNotFoundError as exc:
            self.state.setText(str(exc))
            self.state.set_tone("fail")
            self.state.show()
            return None
        self.state.setText("Evaluating…")
        self.state.set_tone("info")
        self.state.show()
        self.evaluate_button.setEnabled(False)
        return self._job

    def _job_finished(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        self.evaluate_button.setEnabled(True)
        if status == "completed":
            self.state.setText("Done")
            self.state.set_tone("ok")
            split, checkpoint = self.split_combo.currentData(), self.checkpoint_combo.currentData()
            folder = self.ctx.experiments.view(self.run_id).run_dir / "evaluations"
            self._run_chosen(select=folder / f"{split}-{checkpoint}.json")
        else:
            self.state.setText(self.ctx.jobs.job(job_id).error or status.capitalize())
            self.state.set_tone("fail")
