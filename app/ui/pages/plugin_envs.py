"""The Plugin Store's *Environments* card: the optional environment packs (``configs/packs.yaml``),
each a virtual environment layered over the experiment Python, with its libraries, their licences
and whether it is installed. Install and Remove run in the background; jobs that need a pack (its
runners, the inference worker) run under it once it is installed."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from core.common import packs
from core.common.config import experiment_python

from ...services.context import AppContext
from ...services.experiment_setup import run_steps
from ..icons import icon
from ..widgets import Card, DataTable, Pill, label

__all__ = ["EnvironmentsCard"]


class EnvironmentsCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Environments", "heavy libraries in their own environments, installed on request",
                         padded=False, parent=parent)
        self.ctx = ctx
        self.packs = packs.load_packs(ctx.paths.code_root)
        self.names = list(self.packs)
        self._job: str | None = None
        self.table = DataTable(("Pack", "Libraries", "Licences", "State"), mono_columns=(1,),
                               stretch_column=0)
        self.table.itemSelectionChanged.connect(self._update)
        self.add(self.table)
        row = QHBoxLayout()
        self.install_button = QPushButton(icon("package-plus"), "Install")
        self.install_button.setObjectName("Primary")
        self.install_button.clicked.connect(lambda: self.install())
        self.remove_button = QPushButton(icon("trash-2"), "Remove")
        self.remove_button.clicked.connect(lambda: self.remove())
        row.addWidget(self.install_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        self.add(row)
        self.note = label("", "CardCaption", wrap=True)
        self.add(self.note)
        ctx.jobs.job_progress.connect(self._progress)
        ctx.jobs.job_finished.connect(self._finished)
        self.refresh()
        if self.names:
            self.table.selectRow(0)

    @property
    def selected(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        return self.names[rows[0].row()] if rows else None

    def refresh(self) -> None:
        keep = self.selected
        self.table.blockSignals(True)
        self.table.clear_rows()
        for name in self.names:
            pack = self.packs[name]
            facts = packs.describe(pack)
            state = Pill(f"Installed · {facts['size_mb']:.0f} MB" if facts["installed"] else "Not installed",
                         "ok" if facts["installed"] else "planned")
            self.table.add_row((pack.title, ", ".join(pack.requirements),
                                ", ".join(sorted(set(pack.licences.values()))), state))
        self.table.blockSignals(False)
        if keep in self.names:
            self.table.selectRow(self.names.index(keep))
        self._update()

    def _update(self) -> None:
        name = self.selected
        busy = self._job is not None
        installed = name is not None and packs.pack_python(name) is not None
        self.install_button.setEnabled(name is not None and not busy)
        self.install_button.setText("Update" if installed else "Install")
        self.remove_button.setEnabled(installed and not busy)
        if name is not None and not busy:
            pack = self.packs[name]
            self.note.setText(f"{pack.description} Lives in {packs.pack_dir(name)}; it reuses the experiment "
                              "Python's packages, so PyTorch is not installed twice.")

    def install(self) -> str | None:
        name, base = self.selected, experiment_python(self.ctx.config)
        if name is None:
            return None
        if not base:
            self.note.setText("Set up the experiment Python first (Settings → Experiment Python).")
            return None
        pack = self.packs[name]

        def work(cancel: Any, progress: Any) -> str:
            run_steps(packs.install_steps(pack, base), cancel, progress)
            return name

        self._job = self.ctx.jobs.submit_task(work, title=f"Install the {name} environment")
        self._update()
        return self._job

    def remove(self) -> bool:
        name = self.selected
        if name is None or self._job is not None:
            return False
        removed = packs.remove(name)
        self.ctx.inference.stop()  # the next request starts under the right Python
        self.note.setText(f"Removed the {name} environment." if removed else "Nothing to remove.")
        self.refresh()
        return removed

    def _progress(self, job_id: str, _fraction: float, message: str) -> None:
        if job_id == self._job:
            self.note.setText(message)

    def _finished(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        self.note.setText("Installed; jobs that need it now run under it." if status == "completed"
                          else f"Could not install it: {job.error}")
        self.ctx.inference.stop()
        self.refresh()
