"""Studies & Sweeps: groups of runs started from the Experiment Builder's *Repeat and sweep* card.
A seed study shows each metric as a mean with its spread and interval; a sweep shows how its
parameters relate to a score. Both export every number with the run, seed, budget and method
behind it."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QPushButton, QStackedWidget, QWidget

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, Page, Segmented, label
from .seeds import SeedView
from .sweeps import SweepView

__all__ = ["StudiesPage"]


class StudiesPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Studies & Sweeps", "seeds and hyperparameters, with the statistics behind them",
                         parent)
        self.ctx = ctx
        self.switch = Segmented((("seeds", "Seed studies"), ("sweep", "Sweeps")))
        self.switch.changed.connect(lambda _: self.refresh())
        self.head.add_action(self.switch)
        chooser = Card("Group")
        self.group = QComboBox()
        self.group.setMinimumWidth(260)
        self.group.currentIndexChanged.connect(lambda _: self._show())
        chooser.add_head_widget(self.group)
        self.export_button = QPushButton(icon("download"), "Export")
        self.export_button.setToolTip("study.json (every run, statistics and their method), runs.csv and "
                                      "study.md into reports/")
        self.export_button.clicked.connect(lambda: self.export())
        chooser.add_head_widget(self.export_button)
        self.note = label("", "CardCaption", wrap=True)
        chooser.add(self.note)
        self.body.addWidget(chooser)
        self.views = {"seeds": SeedView(ctx), "sweep": SweepView(ctx)}
        self.stack = QStackedWidget()
        for view in self.views.values():
            self.stack.addWidget(view)
        self.body.addWidget(self.stack)
        self.body.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._show)
        ctx.experiments.run_changed.connect(lambda _run: self._timer.start())
        self.refresh()

    @property
    def kind(self) -> str:
        return self.switch.current

    def refresh(self) -> None:
        groups = self.ctx.studies.groups(self.kind)
        chosen = self.group.currentData()
        self.group.blockSignals(True)
        self.group.clear()
        for group in groups:
            title = f"{group['name']} · {group['id'][:6]} · {group['created_at'][:16]}"
            self.group.addItem(title, group["id"])
        self.group.setCurrentIndex(max(self.group.findData(chosen), 0))
        self.group.blockSignals(False)
        self.stack.setCurrentWidget(self.views[self.kind])
        self._show()

    def show_group(self, group_id: str) -> None:
        group = self.ctx.studies.group(group_id)
        if self.kind != group["kind"]:
            self.switch.select(group["kind"])  # refreshes the list
        else:
            self.refresh()
        self.group.setCurrentIndex(max(self.group.findData(group_id), 0))

    def _show(self) -> None:
        if self.ctx.store.closed:  # a run's late change after the app closed its workspace
            return
        group_id = self.group.currentData()
        self.export_button.setEnabled(group_id is not None)
        what = "seed studies" if self.kind == "seeds" else "sweeps"
        self.note.setText("" if group_id else f"No {what} yet: start one from the Experiment Builder's "
                                              "Repeat and sweep card.")
        self.views[self.kind].set_group(group_id)

    def export(self) -> dict:
        group_id = self.group.currentData()
        if group_id is None:
            return {}
        written = self.ctx.studies.export(group_id)
        self.note.setText(f"Exported to {written['json'].parent}")
        return written
