"""Checking for a newer release of the lab on GitHub, in the background. Nothing is downloaded:
the answer names the release and links to its page."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from core.common.updates import RELEASES, is_newer
from core.vcs import GitHub, GitHubError

from .. import __version__
from .jobs import JobQueue

__all__ = ["UpdateCheck", "UpdateResult"]

log = logging.getLogger("tedo.updates")


@dataclass(frozen=True)
class UpdateResult:
    newer: bool
    #: The latest release's tag, name and page; empty when none is published.
    tag: str = ""
    name: str = ""
    url: str = ""
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return f"Could not check for updates: {self.error}"
        if not self.tag:
            return "No release is published yet."
        if self.newer:
            return f"{self.name or self.tag} is available (you have {__version__})."
        return f"You have the latest version ({__version__})."


class UpdateCheck(QObject):
    finished = Signal(object)  # UpdateResult

    def __init__(self, jobs: JobQueue, parent: QObject | None = None, *, api: str = "") -> None:
        super().__init__(parent)
        self.jobs, self.api = jobs, api
        self.result: UpdateResult | None = None
        self._job: str | None = None
        jobs.job_finished.connect(self._done)

    @property
    def checking(self) -> bool:
        return self._job is not None

    def check(self) -> None:
        if self._job is None:
            self._job = self.jobs.submit_task(lambda cancel, progress: self._latest(),
                                              title="Check for updates", quiet=True)

    def _latest(self) -> UpdateResult:
        github = GitHub(api=self.api) if self.api else GitHub()
        try:
            release = github.latest_release(*RELEASES)
        except GitHubError as exc:  # offline, a proxy, rate limits: worth saying, not an error
            log.info("Update check: %s", exc)
            return UpdateResult(False, error=str(exc))
        if release is None:
            return UpdateResult(False)
        return UpdateResult(is_newer(release.tag, __version__), release.tag, release.name, release.html_url)

    def _done(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.jobs.job(job_id)
        self.result = job.result if status == "completed" else UpdateResult(False, error=job.error or status)
        self.finished.emit(self.result)
