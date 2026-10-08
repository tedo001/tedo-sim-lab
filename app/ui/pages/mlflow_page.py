"""MLflow: where runs are tracked, the runs MLflow holds, and the MLflow web UI."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from core.common import distribution_version, mlflow_tracking_uri
from core.common.cancel import CancelToken, ProgressFn

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, Pill, label

__all__ = ["MlflowPage", "read_mlflow_runs"]

_TONES = {"FINISHED": "ok", "RUNNING": "info", "FAILED": "fail", "KILLED": "warn"}
_UI = {"stopped": ("Stopped", "planned"), "starting": ("Starting…", "info"), "running": ("Running", "ok"),
       "failed": ("Failed to start", "fail")}


def read_mlflow_runs(tracking_uri: str, limit: int = 100) -> list[dict[str, Any]]:
    """The newest runs in every MLflow experiment (runs in a background task)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri)
    experiments = {e.experiment_id: e.name for e in client.search_experiments()}
    if not experiments:
        return []
    runs = client.search_runs(list(experiments), max_results=limit, order_by=["attributes.start_time DESC"])
    return [{"experiment": experiments.get(run.info.experiment_id, "?"), "run": run.info.run_id,
             "lab_run": run.data.tags.get("tedo.run_id", ""), "status": run.info.status,
             "started": run.info.start_time, "test_acc": run.data.metrics.get("test_acc"),
             "model": run.data.tags.get("tedo.model", ""), "dataset": run.data.tags.get("tedo.dataset", "")}
            for run in runs]


class MlflowPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("MLflow", "every run is mirrored to MLflow", parent)
        self.ctx = ctx
        self._job: str | None = None
        version = distribution_version("mlflow")
        store = Card("Tracking")
        self.tracking_pill = Pill("On" if ctx.config.mlflow_tracking and version else "Off",
                                  "ok" if ctx.config.mlflow_tracking and version else "warn")
        store.add_head_widget(self.tracking_pill)
        store.add(KeyValues((("Tracking URI", mlflow_tracking_uri(ctx.config, ctx.paths)),
                             ("Artifacts", ctx.paths.mlruns),
                             ("Installed", f"mlflow {version}" if version else "not installed"))))
        store.add(label("Workers log each run's spec, environment snapshot, per-epoch metrics, test results "
                        "and best checkpoint. Switch it off with mlflow_tracking: false in settings.yaml.",
                        "CardCaption", wrap=True))
        self.body.addWidget(store)

        ui = Card("MLflow UI", "MLflow's own web interface, on this computer only")
        self.ui_pill = Pill("", "planned")
        ui.add_head_widget(self.ui_pill)
        row = QHBoxLayout()
        self.ui_button = QPushButton(icon("play"), "Start")
        self.ui_button.clicked.connect(self._toggle_ui)
        self.open_button = QPushButton(icon("chart-line"), "Open in browser")
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ctx.mlflow_ui.url or "")))
        self.url = label("", "Mono")
        for widget in (self.ui_button, self.open_button, self.url):
            row.addWidget(widget)
        row.addStretch(1)
        ui.add(row)
        self.ui_output = label("", "Mono", wrap=True)
        ui.add(self.ui_output)
        self.ui_button.setEnabled(bool(version))
        self.body.addWidget(ui)

        runs = Card("Runs in MLflow", padded=False)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        runs.add_head_widget(self.refresh_button)
        self.table = DataTable(("Experiment", "Lab run", "Data · model", "Status", "Test acc", "Started"),
                               mono_columns=(1, 4), stretch_column=0)
        self.message = label("", "CardCaption", wrap=True)
        self.message.setContentsMargins(14, 10, 14, 12)
        runs.add(self.table)
        runs.add(self.message)
        self.body.addWidget(runs)
        self.body.addStretch(1)

        ctx.mlflow_ui.state_changed.connect(self._ui_state)
        ctx.jobs.job_finished.connect(self._loaded)
        ctx.experiments.run_changed.connect(lambda _: self.refresh())
        self._ui_state(ctx.mlflow_ui.state)
        self.refresh()

    # Runs ----------------------------------------------------------------------------
    def refresh(self) -> None:
        if not distribution_version("mlflow"):
            self.table.hide()
            self.message.setText("MLflow is not installed, so runs are recorded in the lab database only.")
            return
        if self._job is not None:
            return
        uri = mlflow_tracking_uri(self.ctx.config, self.ctx.paths)
        self.message.setText("Reading MLflow…")

        def work(cancel: CancelToken, progress: ProgressFn) -> list[dict[str, Any]]:
            return read_mlflow_runs(uri)

        self._job = self.ctx.jobs.submit_task(work, title="Read MLflow runs", quiet=True)

    def _loaded(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        rows = job.result if status == "completed" else []
        self.show_runs(rows or [], job.error if status == "failed" else None)

    def show_runs(self, rows: list[dict[str, Any]], error: str | None = None) -> None:
        self.table.clear_rows()
        for row in rows:
            started = (datetime.fromtimestamp(row["started"] / 1000).strftime("%d %b %H:%M")
                       if row["started"] else "—")
            accuracy = "—" if row["test_acc"] is None else f"{row['test_acc']:.4f}"
            self.table.add_row((row["experiment"], row["lab_run"][:12] or row["run"][:12],
                                f"{row['dataset']} · {row['model']}" if row["model"] else "—",
                                Pill(row["status"].capitalize(), _TONES.get(row["status"], "planned")),
                                accuracy, started))
        self.table.setVisible(bool(rows))
        self.message.setText(f"Could not read MLflow: {error}" if error else
                             "" if rows else "No runs in MLflow yet: train one and it appears here.")
        self.message.setVisible(bool(self.message.text()))

    # UI ----------------------------------------------------------------------------
    def _toggle_ui(self) -> None:
        if self.ctx.mlflow_ui.state in ("starting", "running"):
            self.ctx.mlflow_ui.stop()
        else:
            self.ctx.mlflow_ui.start()

    def _ui_state(self, state: str) -> None:
        text, tone = _UI.get(state, (state, "planned"))
        self.ui_pill.setText(text)
        self.ui_pill.set_tone(tone)
        running = state in ("starting", "running")
        self.ui_button.setText("Stop" if running else "Start")
        self.ui_button.setIcon(icon("x" if running else "play"))
        self.open_button.setEnabled(state == "running")
        self.url.setText(self.ctx.mlflow_ui.url or "")
        output = self.ctx.mlflow_ui.output.strip().splitlines()
        self.ui_output.setText("\n".join(output[-4:]) if state == "failed" else "")
        self.ui_output.setVisible(state == "failed")
