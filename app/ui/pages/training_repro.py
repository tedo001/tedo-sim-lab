"""Two cards of a run on the Training page: the data checks made before it trained
(``data_validation.json``), and its reproducibility report (question, hypothesis, data provenance,
environment, baseline, results, limitations, the commands that reproduce it) with an export of
the bundle another person reproduces it from."""

from __future__ import annotations

import json

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ...services.context import AppContext
from ...services.reports import export_reproduction, reproduction_report
from ...services.runs import RunView
from ..icons import icon
from ..widgets import Card, label
from ..widgets.markdown import MarkdownView
from .data_checks_view import ChecksTable

__all__ = ["DataChecksCard", "ReproCard"]


class DataChecksCard(Card):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Data checks", "before training: duplicates, leakage, labels, balance, licence",
                         parent=parent)
        self.checks = ChecksTable()
        self.add(self.checks)
        self.summary = label("", "CardCaption", wrap=True)
        self.add(self.summary)

    def show_run(self, view: RunView | None) -> None:
        path = view.run_dir / "data_validation.json" if view is not None else None
        report = None
        if path is not None and path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = None
        self.setVisible(report is not None)
        if report is not None:
            self.checks.show_checks(report.get("checks", []))
            sizes = ", ".join(f"{name} {count:,}" for name, count in report.get("splits", {}).items())
            self.summary.setText(f"{report.get('summary')} · rule: {report.get('rule', 'warn')}"
                                 + (f" · {sizes}" if sizes else "") + (f" · {report['note']}"
                                                                       if report.get("note") else ""))


class ReproCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Reproducibility", "reproduction/report.md", parent=parent)
        self.ctx = ctx
        self.view: RunView | None = None
        self.report = MarkdownView()
        self.report.setMinimumHeight(260)
        self.report.setMaximumHeight(520)
        self.add(self.report)
        buttons = QHBoxLayout()
        self.export_button = QPushButton(icon("package"), "Export bundle")
        self.export_button.setToolTip("The bundle (zip: experiment.yaml, snapshot, package lock, results, "
                                      "report) and the report as PDF, into the workspace's reports/")
        self.export_button.clicked.connect(lambda: self.export())
        self.open_button = QPushButton(icon("folder-open"), "Open report folder")
        self.open_button.clicked.connect(self._open)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        self.add(buttons)
        self.note = label("", "CardCaption", wrap=True)
        self.add(self.note)

    def show_run(self, view: RunView | None) -> None:
        self.view = view
        report = reproduction_report(view, self.ctx.store) if view is not None and view.finished_ok else None
        self.setVisible(report is not None)
        if report is not None:
            self.report.set_markdown(report.read_text(encoding="utf-8"), report.parent)
            self.note.setText("Another person reproduces it from the bundle: python -m "
                              "core.experiment_engine.reproduce <bundle>.zip --workspace <a new folder>")

    def export(self) -> dict:
        if self.view is None:
            return {}
        written = export_reproduction(self.view, self.ctx.store, self.ctx.paths.reports)
        if written:
            self.note.setText(f"Bundle and report written to {written['zip'].parent}")
        return written

    def _open(self) -> None:
        if self.view is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.view.run_dir / "reproduction")))
