"""The cards on Home: active job, recent experiments, job history, catalogue, MLflow."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from PySide6.QtWidgets import QWidget

from core.common import mlflow_tracking_uri
from core.common.optional import distribution_version
from core.experiment_engine.events import ProgressEvent

from ...services.context import AppContext
from ..widgets import Card, DataTable, KeyValues, Pill, label

__all__ = ["ActiveJobCard", "CatalogueCard", "JobHistoryCard", "MlflowCard", "RecentExperimentsCard",
           "local_time"]

_TONES = {"queued": "planned", "running": "info", "completed": "ok", "failed": "fail",
          "cancelled": "warn", "interrupted": "warn"}


def local_time(iso: str | None) -> str:
    """``08 Oct 07:12`` in the person's own time zone."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %H:%M")
    except ValueError:
        return iso


_WORDS = {"catalog_only": "catalogue only", "not_downloaded": "not downloaded",
          "not_installed": "not installed", "not_connected": "not connected"}


def _counts(statuses: list[str]) -> str:
    counts = Counter(statuses)
    return " · ".join(f"{n} {_WORDS.get(status, status)}" for status, n in counts.most_common())


class ActiveJobCard(Card):
    """The job running now and its latest progress, live from the queue."""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Active job", parent=parent)
        self.jobs = ctx.jobs
        self.pill = Pill("Idle", "planned")
        self.add_head_widget(self.pill)
        self.title = label("No job is running.", "BodyStrong", wrap=True)
        self.detail = label("Start one from the Experiment Builder or a Computer Vision preset.",
                            "CardCaption", wrap=True)
        self.metrics = label("", "Mono", wrap=True)
        for widget in (self.title, self.detail, self.metrics):
            self.add(widget)
        self._job_id: str | None = None
        self.jobs.job_started.connect(self._started)
        self.jobs.job_event.connect(self._event)
        self.jobs.job_finished.connect(self._finished)
        running = [job for job in self.jobs.active() if job.status == "running"]
        if running:
            self._started(running[-1].id)

    def _started(self, job_id: str) -> None:
        job = self.jobs.job(job_id)
        if job.quiet:
            return
        self._job_id = job_id
        self.pill.setText("Running")
        self.pill.set_tone("info")
        self.title.setText(job.title)
        self.detail.setText("Starting…" if job.kind == "run" else "Background task")
        self.metrics.setText("")

    def _event(self, job_id: str, event: ProgressEvent) -> None:
        if job_id != self._job_id:
            return
        payload = event.payload
        if event.type in ("epoch_start", "epoch_end"):
            total = payload.get("total")
            self.detail.setText(f"Epoch {payload.get('epoch')}" + (f" of {total}" if total else ""))
        if event.type in ("step", "epoch_end") and payload.get("metrics"):
            self.metrics.setText("  ".join(f"{key} {value:.4g}" for key, value
                                           in payload["metrics"].items()))

    def _finished(self, job_id: str, status: str) -> None:
        if job_id != self._job_id:
            return
        self.pill.setText(status.capitalize())
        self.pill.set_tone(_TONES.get(status, "planned"))
        error = self.jobs.job(job_id).error
        self.detail.setText(error or "Finished.")
        self._job_id = None


class RecentExperimentsCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Recent experiments", padded=False, parent=parent)
        self.store = ctx.store
        self.table = DataTable(("Experiment", "Task", "Created"), stretch_column=0)
        self.empty = label("No experiments yet. Describe one in the Experiment Builder.",
                           "CardCaption", wrap=True)
        self.empty.setContentsMargins(14, 10, 14, 12)
        self.add(self.table)
        self.add(self.empty)
        self.refresh()

    def refresh(self) -> None:
        rows = self.store.experiments(limit=6)
        self.table.clear_rows()
        for row in rows:
            self.table.add_row((row["name"], row["task"].replace("_", " "), local_time(row["created_at"])))
        self.table.setVisible(bool(rows))
        self.empty.setVisible(not rows)


class JobHistoryCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Job history", "last six", padded=False, parent=parent)
        self.store = ctx.store
        self.table = DataTable(("Job", "Kind", "Status", "Ended"), stretch_column=0)
        self.empty = label("No jobs have run in this workspace yet.", "CardCaption", wrap=True)
        self.empty.setContentsMargins(14, 10, 14, 12)
        self.add(self.table)
        self.add(self.empty)
        ctx.jobs.job_finished.connect(lambda *_: self.refresh())
        ctx.jobs.job_queued.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        rows = self.store.jobs(limit=6)
        self.table.clear_rows()
        for row in rows:
            self.table.add_row((row["title"], row["kind"],
                                Pill(row["status"].capitalize(), _TONES.get(row["status"], "planned")),
                                local_time(row["ended_at"])))
        self.table.setVisible(bool(rows))
        self.empty.setVisible(not rows)


class CatalogueCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Catalogue", "what the lab knows about", parent=parent)
        catalog = ctx.catalog
        datasets = [catalog.datasets.status(card.id) for card in catalog.datasets.all()]
        models = [catalog.models.status(card.id) for card in catalog.models.all()]
        plugins = [catalog.plugins.status(m.name) for m in catalog.plugins.manifests()]
        runners = [runner.maturity for runner in catalog.runners.all()]
        self.values = KeyValues((
            ("Datasets", f"{len(datasets)}: {_counts(datasets)}"),
            ("Models", f"{len(models)}: {_counts(models)}"),
            ("Plugins", f"{len(plugins)}: {_counts(plugins)}"),
            ("Runners", f"{len(runners)}: {_counts(runners)}"),
        ), mono=False)
        self.add(self.values)
        self.add(label("Licences: only permissively licensed tools (MIT, Apache-2.0, BSD).",
                       "CardCaption", wrap=True))


class MlflowCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("MLflow", parent=parent)
        version = distribution_version("mlflow")
        tracking = ctx.config.mlflow_tracking and bool(distribution_version("mlflow"))
        self.pill = Pill("Tracking every run" if tracking else "Off", "ok" if tracking else "warn")
        self.add_head_widget(self.pill)
        self.values = KeyValues((
            ("Tracking URI", mlflow_tracking_uri(ctx.config, ctx.paths)),
            ("Artifacts", ctx.paths.mlruns),
            ("Installed", f"mlflow {version}" if version else "not installed"),
        ))
        self.add(self.values)
