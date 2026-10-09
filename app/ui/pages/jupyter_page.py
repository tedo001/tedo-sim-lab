"""Jupyter Notebook: the workspace's notebooks (new, started from a run, linked to runs, read in
place) and Jupyter Lab, installed on request into the experiment Python and opened in the
browser with a fresh token. Code runs in Jupyter Lab; the lab itself never executes a cell."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QPushButton, QWidget

from core.common import experiment_python
from core.common.masking import mask_text

from ...services.context import AppContext
from ...services.jupyter import JUPYTER_REQUIREMENTS, jupyter_version, notebook_markdown
from ..icons import icon
from ..widgets import Card, DataTable, Page, Pill, label
from ..widgets.markdown import MarkdownView

__all__ = ["JupyterPage"]

_STATE = {"stopped": ("Stopped", "planned"), "starting": ("Starting", "info"), "running": ("Running", "ok"),
          "failed": ("Failed", "fail")}


class JupyterPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Jupyter Notebook", "notebooks in this workspace, and Jupyter Lab", parent)
        self.ctx = ctx
        self.version: str | None = None
        self.files = []
        self._check_job = self._install_job = None

        lab = Card("Jupyter Lab", "runs in the experiment Python; bound to this computer only")
        row = QHBoxLayout()
        self.state = Pill("Checking…", "planned")
        row.addWidget(self.state)
        self.version_text = label("", "Mono")
        row.addWidget(self.version_text, 1)
        self.install_button = QPushButton(icon("package-plus"), "Install Jupyter Lab")
        self.install_button.clicked.connect(self.install)
        self.start_button = QPushButton(icon("play"), "Start")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(ctx.jupyter.start)
        self.stop_button = QPushButton(icon("square"), "Stop")
        self.stop_button.clicked.connect(ctx.jupyter.stop)
        self.browser_button = QPushButton(icon("external-link"), "Open in browser")
        self.browser_button.clicked.connect(lambda: self._browse(None))
        for button in (self.install_button, self.start_button, self.stop_button, self.browser_button):
            row.addWidget(button)
        lab.add(row)
        self.lab_text = label("", "Mono", wrap=True)
        lab.add(self.lab_text)
        lab.add(label(f"Installing runs pip install {' '.join(JUPYTER_REQUIREMENTS)} (BSD-3-Clause) in "
                      f"{experiment_python(ctx.config)}.", "CardCaption", wrap=True))
        self.body.addWidget(lab)

        books = Card("Notebooks", "in this workspace's notebooks/ folder", padded=False)
        self.table = DataTable(("Notebook", "Changed", "Runs"), mono_columns=(1, 2), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._picked)
        books.add(self.table)
        self.body.addWidget(books)

        new = Card("New notebook", "optionally started from a run: it reads that run's files")
        row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name")
        row.addWidget(self.name, 1)
        self.run_choice = QComboBox()
        row.addWidget(self.run_choice, 1)
        self.create_button = QPushButton(icon("file-plus"), "Create")
        self.create_button.clicked.connect(self.create)
        row.addWidget(self.create_button)
        new.add(row)
        self.body.addWidget(new)

        self.viewer_card = Card("Notebook")
        row = QHBoxLayout()
        self.open_lab_button = QPushButton(icon("external-link"), "Open in Jupyter Lab")
        self.open_lab_button.clicked.connect(lambda: self._browse(self.current()))
        self.open_file_button = QPushButton(icon("folder-open"), "Open file")
        self.open_file_button.clicked.connect(self._open_file)
        self.link_button = QPushButton(icon("tag"), "Link to the chosen run")
        self.link_button.clicked.connect(self.link)
        for button in (self.open_lab_button, self.open_file_button, self.link_button):
            row.addWidget(button)
        row.addStretch(1)
        self.viewer_card.add(row)
        self.viewer = MarkdownView()
        self.viewer.setMinimumHeight(360)
        self.viewer_card.add(self.viewer)
        self.body.addWidget(self.viewer_card)
        self.body.addStretch(1)

        ctx.jupyter.state_changed.connect(lambda _: self._update())
        ctx.jobs.job_finished.connect(self._job_done)
        ctx.experiments.run_changed.connect(lambda _: self._fill_runs())
        self._fill_runs()
        self.refresh()
        self.check()

    # Jupyter Lab ---------------------------------------------------------------------------------
    def check(self) -> None:
        python = experiment_python(self.ctx.config)
        self._check_job = self.ctx.jobs.submit_task(lambda cancel, progress: jupyter_version(python),
                                                    title="Check Jupyter Lab", quiet=True)
        self._update()

    def install(self) -> str | None:
        try:
            command = self.ctx.jupyter.install_command()
        except RuntimeError as exc:
            self.lab_text.setText(str(exc))
            return None

        def work(cancel, progress) -> None:
            result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            if result.returncode != 0:
                tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-12:])
                raise RuntimeError(mask_text(f"pip failed ({result.returncode}):\n{tail}"))

        self.lab_text.setText("Installing Jupyter Lab… this takes a minute or two.")
        self._install_job = self.ctx.jobs.submit_task(work, title="Install Jupyter Lab")
        self._update()
        return self._install_job

    def _job_done(self, job_id: str, status: str) -> None:
        if job_id == self._check_job:
            self._check_job = None
            self.version = self.ctx.jobs.job(job_id).result if status == "completed" else None
        elif job_id == self._install_job:
            self._install_job = None
            error = self.ctx.jobs.job(job_id).error
            self.lab_text.setText(f"Install failed: {error}" if status != "completed" else "Installed.")
            self.check()
        else:
            return
        self._update()

    def _update(self) -> None:
        lab = self.ctx.jupyter
        checking = self._check_job is not None
        words, tone = ("Checking…", "planned") if checking and self.version is None else (
            _STATE[lab.state] if self.version else ("Not installed", "warn"))
        self.state.setText(words)
        self.state.set_tone(tone)
        self.version_text.setText(f"jupyterlab {self.version}" if self.version else "")
        installing = self._install_job is not None
        self.install_button.setVisible(not self.version and not checking)
        self.install_button.setEnabled(not installing)
        self.start_button.setEnabled(bool(self.version) and lab.state in ("stopped", "failed"))
        self.stop_button.setEnabled(lab.state in ("starting", "running"))
        self.browser_button.setEnabled(lab.state == "running")
        self.open_lab_button.setEnabled(lab.state == "running" and self.current() is not None)
        if lab.state == "running":
            self.lab_text.setText(f"Running at {lab.url} (the access token is added when you open it).")
        elif lab.state == "failed":
            self.lab_text.setText(mask_text(lab.output[-1500:]) or "Jupyter Lab did not start.")

    def _browse(self, notebook: Path | None) -> None:
        link = self.ctx.jupyter.link_for(notebook)
        if link:
            QDesktopServices.openUrl(QUrl(link))

    # Notebooks -----------------------------------------------------------------------------------
    def _fill_runs(self) -> None:
        keep = self.run_choice.currentData()
        self.run_choice.clear()
        self.run_choice.addItem("Not from a run", None)
        for view in self.ctx.experiments.views(limit=100):
            if view.finished_ok:
                self.run_choice.addItem(f"{view.name} · {view.id[:8]}", view.id)
        index = self.run_choice.findData(keep)
        self.run_choice.setCurrentIndex(max(index, 0))

    def refresh(self, select: Path | None = None) -> None:
        self.files = self.ctx.notebooks.list()
        self.table.clear_rows()
        for notebook in self.files:
            changed = datetime.fromtimestamp(notebook.modified).strftime("%Y-%m-%d %H:%M")
            self.table.add_row([notebook.relative, changed, ", ".join(r[:8] for r in notebook.runs) or "—"])
        paths = [notebook.path for notebook in self.files]
        if select in paths:
            self.table.selectRow(paths.index(select))
        elif self.files:
            self.table.selectRow(0)
        self._picked()

    def current(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows()
        return self.files[rows[0].row()].path if rows and rows[0].row() < len(self.files) else None

    def _picked(self) -> None:
        path = self.current()
        self.viewer_card.setVisible(path is not None)
        if path is not None:
            self.viewer_card.set_title(path.name)
            self.viewer.set_markdown(notebook_markdown(path))
        self.open_file_button.setEnabled(path is not None)
        self.link_button.setEnabled(path is not None)
        self._update()

    def create(self) -> Path:
        run_id = self.run_choice.currentData()
        view = self.ctx.experiments.view(run_id) if run_id else None
        path = self.ctx.notebooks.create(self.name.text().strip(), view)
        self.name.clear()
        self.refresh(select=path)
        return path

    def link(self) -> None:
        path, run_id = self.current(), self.run_choice.currentData()
        if path is None or run_id is None:
            self.lab_text.setText("Choose a run in New notebook first, then link.")
            return
        self.ctx.notebooks.link(path, self.ctx.experiments.view(run_id))
        self.refresh(select=path)

    def _open_file(self) -> None:
        path = self.current()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
