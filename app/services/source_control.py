"""Git and GitHub for the Source Control page. Every operation runs as a quiet background task
(git can take a while, the network longer); the result, or why it failed, comes back to a
callback on the GUI thread. The GitHub token comes from the OS keyring or ``GITHUB_TOKEN``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.common import AppPaths, CredentialStore
from core.vcs import Git, GitHub, git_executable
from core.vcs.github import API

from .jobs import JobQueue

__all__ = ["TOKEN_KEY", "SourceControl"]

TOKEN_KEY = "GITHUB_TOKEN"


class SourceControl(QObject):
    #: The repository changed (a commit, a checkout, a pull): views refresh.
    changed = Signal()
    #: The GitHub account was connected or disconnected.
    account_changed = Signal()

    def __init__(self, paths: AppPaths, credentials: CredentialStore, jobs: JobQueue,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.credentials, self.jobs = paths, credentials, jobs
        self.root: Path | None = None
        self.api = API  # tests point this at a local stand-in
        self._callbacks: dict[str, tuple[Callable[[Any], None], Callable[[str], None] | None]] = {}
        jobs.job_finished.connect(self._finished)

    @property
    def git_available(self) -> bool:
        return git_executable() is not None

    def token(self) -> str:
        secret = self.credentials.get(TOKEN_KEY)
        return secret.reveal() if secret else ""

    def token_source(self) -> str:
        return self.credentials.source(TOKEN_KEY)

    def git(self, root: Path | None = None) -> Git:
        return Git(root or self.root or self.paths.workspace, token=self.token())

    def github(self, token: str | None = None) -> GitHub:
        return GitHub(self.token() if token is None else token, api=self.api)

    def run(self, title: str, work: Callable[[], Any], done: Callable[[Any], None],
            failed: Callable[[str], None] | None = None) -> str:
        """Run ``work()`` in the background; then ``done(result)`` or ``failed(message)``."""
        job_id = self.jobs.submit_task(lambda cancel, progress: work(), title=title, quiet=True)
        self._callbacks[job_id] = (done, failed)
        return job_id

    def _finished(self, job_id: str, status: str) -> None:
        callbacks = self._callbacks.pop(job_id, None)
        if callbacks is None:
            return
        done, failed = callbacks
        job = self.jobs.job(job_id)
        if status == "completed":
            done(job.result)
        elif failed is not None:
            failed(_short(job.error or status))

    def busy(self) -> bool:
        return bool(self._callbacks)

    # The account -------------------------------------------------------------------------------
    def store_token(self, token: str) -> None:
        self.credentials.set(TOKEN_KEY, token)
        self.account_changed.emit()

    def forget_token(self) -> None:
        self.credentials.delete(TOKEN_KEY)
        self.account_changed.emit()


def _short(message: str) -> str:
    """A task's error without the exception class in front ("GitError: …" → "…")."""
    head, _, rest = message.partition(": ")
    return rest if rest and head.endswith(("Error", "Exception")) else message
