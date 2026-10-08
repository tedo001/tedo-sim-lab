"""Dataset downloads, started by a person and run as background tasks.

One download per dataset at a time, whichever page starts it; every page sees its
progress. The adapter enforces the licence policy: a dataset whose terms need
acknowledging only downloads with ``acknowledged=True``.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from core.common.cancel import CancelToken, ProgressFn
from core.common.licensing import download_policy
from core.dataset_registry import DatasetRegistry

from .jobs import JobQueue

__all__ = ["DownloadService"]


class DownloadService(QObject):
    progress = Signal(str, float, str)    # card id, fraction (< 0 unknown), message
    finished = Signal(str, str, str)      # card id, status, error ("" if none)

    def __init__(self, datasets: DatasetRegistry, jobs: JobQueue, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.datasets, self.jobs = datasets, jobs
        self._jobs: dict[str, str] = {}   # card id → job id
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
        card_id = self._card_of(job_id)
        if card_id is not None:
            del self._jobs[card_id]
            self.finished.emit(card_id, status, self.jobs.job(job_id).error or "")
