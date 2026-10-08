"""One run on the Training page: what it is, how far it got, its curves, results and log."""

from __future__ import annotations

import json
from datetime import datetime

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...services.context import AppContext
from ...services.runs import RunView
from ..icons import icon
from ..widgets import Card, EpochChart, KeyValues, Pill, label
from .home_cards import local_time

__all__ = ["CURVES", "RunDetail", "STATUS_TONES", "duration_text"]

STATUS_TONES = {"queued": "planned", "running": "info", "completed": "ok", "early_stopped": "ok",
                "failed": "fail", "cancelled": "warn", "interrupted": "warn"}
#: (metric, title, is a 0–1 share)
CURVES = (("train_loss", "Training loss", False), ("val_loss", "Validation loss", False),
          ("train_acc", "Training accuracy", True), ("val_acc", "Validation accuracy", True))
_RESULTS = (("test_acc", "Test accuracy"), ("test_f1", "Test F1 (macro)"),
            ("test_precision", "Test precision"),
            ("test_recall", "Test recall"), ("best_epoch", "Best epoch"))


def duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} min {seconds:02d} s" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def status_text(status: str) -> str:
    return status.replace("_", " ").capitalize()


class RunDetail(QWidget):
    """Cards for one run. Call :meth:`show_run` to switch; it follows the run's live signals."""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.view: RunView | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.summary = Card("Run")
        self.status = Pill("", "planned")
        self.summary.add_head_widget(self.status)
        self.values = KeyValues((("Dataset · model", ""), ("Device", ""), ("Started", ""), ("Duration", ""),
                                 ("Run folder", ""), ("Code", ""), ("MLflow run", "")))
        self.summary.add(self.values)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress_text = label("", "Mono", wrap=True)
        self.summary.add(self.progress)
        self.summary.add(self.progress_text)
        self.error = label("", "Body", wrap=True)
        self.summary.add(self.error)
        self.lineage = label("", "Body", wrap=True)
        self.summary.add(self.lineage)
        buttons = QHBoxLayout()
        self.cancel_button = QPushButton(icon("x"), "Cancel")
        self.cancel_button.clicked.connect(lambda: self.view and ctx.experiments.cancel(self.view.id))
        self.resume_button = QPushButton(icon("play"), "Resume")
        self.resume_button.setToolTip("Continue from checkpoints/last.pt")
        self.resume_button.clicked.connect(self._resume)
        self.folder_button = QPushButton(icon("folder-open"), "Open run folder")
        self.folder_button.clicked.connect(lambda: self.view and QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.view.run_dir))))
        self.reproduce_button = QPushButton(icon("git-compare"), "Reproduce")
        self.reproduce_button.setToolTip("Run the same experiment.yaml again and compare the results")
        self.reproduce_button.clicked.connect(self._reproduce)
        self.again_button = QPushButton(icon("sliders-horizontal"), "Edit as new")
        self.again_button.setToolTip("Open this experiment in the Experiment Builder")
        self.again_button.clicked.connect(self._edit)
        self.explain_button = QPushButton(icon("brain-circuit"), "Open in CNN Explainer")
        self.explain_button.clicked.connect(lambda: self.view and ctx.experiments.explain(self.view.id,
                                                                                        ctx.navigate))
        for button in (self.cancel_button, self.resume_button, self.reproduce_button, self.folder_button,
                       self.again_button, self.explain_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.summary.add(buttons)
        layout.addWidget(self.summary)

        curves = Card("Curves", "per epoch")
        grid = QGridLayout()
        grid.setSpacing(16)
        self.charts: dict[str, EpochChart] = {}
        for index, (key, title, share) in enumerate(CURVES):
            chart = EpochChart(title, percent=share)
            self.charts[key] = chart
            grid.addWidget(chart, index // 2, index % 2)
        curves.add(grid)
        layout.addWidget(curves)

        self.results = Card("Test results", "best checkpoint on the held-out test split")
        self.result_values = KeyValues(tuple((title, "—") for _, title in _RESULTS))
        self.results.add(self.result_values)
        layout.addWidget(self.results)

        log = Card("Log", "run.log")
        self.log = QPlainTextEdit()
        self.log.setObjectName("Code")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(180)
        log.add(self.log)
        layout.addWidget(log)

        service = ctx.experiments
        service.run_changed.connect(self._changed)
        service.run_step.connect(self._step)
        service.run_epoch.connect(self._epoch)
        service.run_log.connect(self._log_line)

    # Showing a run ---------------------------------------------------------------
    def show_run(self, run_id: str | None) -> None:
        self.setVisible(run_id is not None)
        if run_id is None:
            self.view = None
            return
        self.view = self.ctx.experiments.view(run_id)
        self.log.setPlainText("\n".join(self.ctx.experiments.log_lines(run_id)))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        self._refresh()

    def _refresh(self) -> None:
        view = self.view
        if view is None:
            return
        self.summary.title.setText(view.name)
        self.status.setText(status_text(view.status))
        self.status.set_tone(STATUS_TONES.get(view.status, "planned"))
        self.values.set_value("Dataset · model", f"{view.dataset} · {view.model}")
        self.values.set_value("Device", view.device or "—")
        self.values.set_value("Started", local_time(view.started_at))
        running = view.status == "running" and view.started_at
        seconds = ((datetime.now().astimezone() - datetime.fromisoformat(view.started_at)).total_seconds()
                   if running else view.duration_s)
        self.values.set_value("Duration", duration_text(seconds))
        self.values.set_value("Run folder", str(view.run_dir))
        self.values.set_value("Code", (view.git_commit or "not a git checkout")[:12])
        self.values.set_value("MLflow run", view.mlflow_run_id or "not tracked")
        self.lineage.setText(self._lineage(view))
        self.lineage.setVisible(bool(self.lineage.text()))
        self.reproduce_button.setVisible(view.finished_ok)
        self.error.setText(view.error or "")
        self.error.setVisible(bool(view.error))
        self.cancel_button.setVisible(view.active)
        self.resume_button.setVisible(view.resumable)
        self.explain_button.setVisible(view.status in ("completed", "early_stopped")
                                       and (view.run_dir / "explainer.npz").is_file())
        history = self.ctx.experiments.history(view.id, tuple(key for key, _, _ in CURVES))
        for key, chart in self.charts.items():
            epochs = view.epochs or 0
            chart.set_points([(epoch, value) for epoch, value in history[key] if epoch <= epochs], epochs)
        metrics = self.ctx.store.latest_metrics(view.id)
        for key, title in _RESULTS:
            value = metrics.get(key)
            text = "—" if value is None else f"{int(value)}" if key == "best_epoch" else f"{value:.4f}"
            self.result_values.set_value(title, text)
        self.results.setVisible(any(key in metrics for key, _ in _RESULTS[:1]))
        self._step(view.id)

    def _step(self, run_id: str) -> None:
        if self.view is None or run_id != self.view.id:
            return
        live = self.ctx.experiments.live(run_id)
        active = self.view.active
        self.progress.setVisible(active)
        if live is None or not active:
            self.progress_text.setText("Waiting in the queue…" if self.view.status == "queued" else "")
            self.progress_text.setVisible(self.view.status == "queued")
            return
        self.progress_text.setVisible(True)
        self.progress.setValue(int(live.fraction * 1000))
        parts = [f"epoch {live.epoch} of {live.epochs}" if live.epoch else "starting…"]
        if live.steps:
            parts.append(f"batch {live.step} of {live.steps}")
        for key in ("train_loss", "train_acc"):
            if key in live.metrics:
                parts.append(f"{key} {live.metrics[key]:.4f}")
        if live.eta_s is not None:
            parts.append(f"about {duration_text(live.eta_s)} of training left")
        self.progress_text.setText(" · ".join(parts))

    def _epoch(self, run_id: str, epoch: int, metrics: dict) -> None:
        if self.view is not None and run_id == self.view.id:
            self._refresh()

    def _changed(self, run_id: str) -> None:
        if self.view is not None and run_id == self.view.id:
            self.view = self.ctx.experiments.view(run_id)
            self._refresh()

    def _log_line(self, run_id: str, line: str) -> None:
        if self.view is not None and run_id == self.view.id:
            self.log.appendPlainText(line)

    def _lineage(self, view: RunView) -> str:
        """For a reproduction: what differed before it ran and how its results compare."""
        if not view.parent_run_id:
            return ""
        lines = [f"Reproduction of run {view.parent_run_id}."]
        record = view.run_dir / "reproduces.json"
        if record.is_file():
            notes = json.loads(record.read_text(encoding="utf-8")).get("notes", [])
            lines += [f"– {note}" for note in notes] or ["– same Python, packages and code as the original"]
        mine = self.ctx.store.latest_metrics(view.id)
        theirs = self.ctx.store.latest_metrics(view.parent_run_id)
        for key in ("test_acc", "test_f1"):
            if key in mine and key in theirs:
                lines.append(f"{key}: {mine[key]:.4f} now, {theirs[key]:.4f} originally "
                             f"(difference {mine[key] - theirs[key]:+.4f})")
        return "\n".join(lines)

    def _reproduce(self) -> None:
        if self.view is not None and self.view.finished_ok:
            self.ctx.experiments.reproduce(self.view.id)

    def _resume(self) -> None:
        if self.view is not None and self.view.resumable:
            self.ctx.experiments.resume(self.view.id)

    def _edit(self) -> None:
        if self.view is not None and self.view.spec is not None:
            self.ctx.experiments.open_in_builder(self.view.spec, self.ctx.navigate)
