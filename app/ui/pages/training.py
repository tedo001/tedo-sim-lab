"""Training: every run in this workspace, newest first, and the selected one in detail."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, Page, Pill, label
from .home_cards import local_time
from .training_run import STATUS_TONES, RunDetail, status_text

__all__ = ["TrainingPage"]


class TrainingPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Training", "queued, running and finished runs", parent)
        self.ctx = ctx
        self.run_ids: list[str] = []
        self.selected: str | None = None
        new = QPushButton(icon("sliders-horizontal"), "New experiment")
        new.clicked.connect(lambda: ctx.navigate("experiment_builder"))
        self.head.add_action(new)

        runs = Card("Runs", padded=False)
        self.table = DataTable(("Experiment", "Data · model", "Status", "Progress", "Best val acc",
                                "Started"),
                               mono_columns=(3, 4), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.empty = label("No runs yet. Describe one in the Experiment Builder, or start from a preset "
                           "in the Computer Vision lab.", "CardCaption", wrap=True)
        self.empty.setContentsMargins(14, 10, 14, 12)
        runs.add(self.table)
        runs.add(self.empty)
        self.body.addWidget(runs)
        self.detail = RunDetail(ctx)
        self.body.addWidget(self.detail)
        self.body.addStretch(1)

        ctx.experiments.run_changed.connect(self._run_changed)
        ctx.experiments.run_epoch.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        views = self.ctx.experiments.views(limit=50)
        self.run_ids = [view.id for view in views]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for view in views:
            history = self.ctx.experiments.history(view.id, ("val_acc", "train_loss"))
            done = len(history["train_loss"])
            best = max((value for _, value in history["val_acc"]), default=None)
            self.table.add_row((view.name, f"{view.dataset} · {view.model}",
                                Pill(status_text(view.status), STATUS_TONES.get(view.status, "planned")),
                                f"{done}/{view.epochs or '?'}", "—" if best is None else f"{best:.2%}",
                                local_time(view.started_at)))
        self.table.blockSignals(False)
        self.table.setVisible(bool(views))
        self.empty.setVisible(not views)
        if self.selected not in self.run_ids:
            self.selected = self.run_ids[0] if self.run_ids else None
        self._highlight()
        self.detail.show_run(self.selected)

    def select(self, run_id: str) -> None:
        self.selected = run_id
        self._highlight()
        self.detail.show_run(run_id)

    def _highlight(self) -> None:
        if self.selected in self.run_ids:
            self.table.blockSignals(True)
            self.table.selectRow(self.run_ids.index(self.selected))
            self.table.blockSignals(False)

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.select(self.run_ids[rows[0].row()])

    def _run_changed(self, run_id: str) -> None:
        known = run_id in self.run_ids
        if not known:  # a new run: follow it
            self.selected = run_id
        self.refresh()
