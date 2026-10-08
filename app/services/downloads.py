"""Dataset downloads and CSV imports, started by a person and run as background tasks.

One download per dataset at a time, whichever page starts it; every page sees its
progress. The adapter enforces the licence policy: a dataset whose terms need
acknowledging only downloads with ``acknowledged=True``. An imported CSV file is copied
into the workspace with a card of its own and joins the catalogue straight away.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.catalog import IMPORTED
from core.common.cancel import CancelToken, ProgressFn
from core.common.licensing import download_policy
from core.dataset_registry import DatasetRegistry

from .jobs import JobQueue

__all__ = ["DownloadService"]


class DownloadService(QObject):
    progress = Signal(str, float, str)    # card id, fraction (< 0 unknown), message
    finished = Signal(str, str, str)      # card id, status, error ("" if none)
    imported = Signal(str, str, str)      # job id, new card id ("" if it failed), error

    def __init__(self, datasets: DatasetRegistry, jobs: JobQueue, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.datasets, self.jobs = datasets, jobs
        self._jobs: dict[str, str] = {}   # card id → job id
        self._imports: set[str] = set()   # job ids of CSV imports
        jobs.job_progress.connect(self._progress)
        jobs.job_finished.connect(self._finished)

    def acknowledgement(self, card_id: str) -> str | None:
        """The terms a person must accept before this download, if any."""
        return download_policy(self.datasets.get(card_id)).acknowledgement

    def refusal(self, card_id: str) -> str | None:
        """Why the lab will not download this dataset, or ``None`` if it may."""
        decision = download_policy(self.datasets.get(card_id))
        return None if decision.allowed else decision.reason

    def active(self, card_id: str) -> bool:
        return card_id in self._jobs

    def start(self, card_id: str, *, acknowledged: bool = False) -> str:
        if card_id in self._jobs:
            return self._jobs[card_id]
        adapter = self.datasets.adapter(card_id)
        root = self.datasets.root

        def work(cancel: CancelToken, progress: ProgressFn) -> None:
            adapter.prepare(root, progress, cancel, acknowledged=acknowledged)

        job_id = self.jobs.submit_task(work, title=f"Download {adapter.card.name}")
        self._jobs[card_id] = job_id
        return job_id

    def import_csv(self, path: Path, *, name: str, target: str | None, task: str) -> str:
        """Copy ``path`` into ``datasets/imported/`` with a card; ``imported`` reports the card id."""
        from labs.classical_ml.datasets import import_csv

        root = self.datasets.root

        def work(cancel: CancelToken, progress: ProgressFn) -> str:
            progress(-1.0, f"Importing {path.name}")
            return import_csv(path, root, name=name, target=target, task=task)

        job_id = self.jobs.submit_task(work, title=f"Import {path.name}")
        self._imports.add(job_id)
        return job_id

    def cancel(self, card_id: str) -> None:
        if card_id in self._jobs:
            self.jobs.cancel(self._jobs[card_id])

    def _card_of(self, job_id: str) -> str | None:
        return next((card for card, job in self._jobs.items() if job == job_id), None)

    def _progress(self, job_id: str, fraction: float, message: str) -> None:
        card_id = self._card_of(job_id)
        if card_id is not None:
            self.progress.emit(card_id, fraction, message)

    def _finished(self, job_id: str, status: str) -> None:
        if job_id in self._imports:
            self._imports.discard(job_id)
            job = self.jobs.job(job_id)
            card_id = job.result if status == "completed" else ""
            if card_id:
                self.datasets.load_file(self.datasets.root / IMPORTED / f"{card_id}.yaml")
            self.imported.emit(job_id, card_id or "", job.error or "")
            return
        card_id = self._card_of(job_id)
        if card_id is not None:
            del self._jobs[card_id]
            self.finished.emit(card_id, status, self.jobs.job(job_id).error or "")
