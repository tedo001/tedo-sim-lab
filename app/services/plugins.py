"""Plugins from the interface's side: install, test a connection, search a source and download
from it as background tasks (nothing here blocks the UI), and keep account credentials in the OS
keyring. Every outcome comes back as one ``finished`` signal."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.catalog import Catalog
from core.common import AppPaths, CredentialStore
from core.common.secrets import CredentialError
from core.plugin_api import Plugin
from labs.common.remote import RemoteItem

from .jobs import JobQueue

__all__ = ["PluginService"]


class PluginService(QObject):
    #: job id, plugin name, action ("install", "test", "search", "download"), result, error ("" = ok)
    finished = Signal(str, str, str, object, str)
    #: plugin name: its status may have changed (installed, credential set or removed)
    changed = Signal(str)

    def __init__(self, paths: AppPaths, catalog: Catalog, credentials: CredentialStore, jobs: JobQueue,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.catalog, self.credentials, self.jobs = paths, catalog, credentials, jobs
        self._jobs: dict[str, tuple[str, str]] = {}
        jobs.job_finished.connect(self._done)

    def plugin(self, name: str) -> Plugin:
        return self.catalog.plugins.get(name)

    def busy(self, name: str, action: str) -> bool:
        return (name, action) in self._jobs.values()

    def _submit(self, name: str, action: str, title: str, work: Callable[[Any, Any], Any]) -> str:
        job_id = self.jobs.submit_task(work, title=title)
        self._jobs[job_id] = (name, action)
        return job_id

    # Actions ------------------------------------------------------------------------------
    def install(self, name: str) -> str:
        plugin = self.plugin(name)
        return self._submit(name, "install", f"Install {plugin.manifest.title}",
                            lambda cancel, progress: plugin.install())

    def test(self, name: str) -> str:
        plugin = self.plugin(name)
        return self._submit(name, "test", f"Test {plugin.manifest.title} connection",
                            lambda cancel, progress: plugin.test_connection())

    def search(self, name: str, query: str) -> str:
        plugin = self.plugin(name)
        return self._submit(name, "search", f"Search {plugin.manifest.title}",
                            lambda cancel, progress: plugin.run("search", query=query))

    def download(self, name: str, item: RemoteItem) -> str:
        plugin = self.plugin(name)
        return self._submit(name, "download", f"Download {item.title} from {plugin.manifest.title}",
                            lambda cancel, progress: plugin.run("download", item=item,
                                                                workspace=self.paths.workspace,
                                                                cancel=cancel, progress=progress))

    def _done(self, job_id: str, status: str) -> None:
        if job_id not in self._jobs:
            return
        name, action = self._jobs.pop(job_id)
        job = self.jobs.job(job_id)
        error = "" if status == "completed" else (job.error or status)
        if action == "install":
            importlib.invalidate_caches()  # new distributions must show up in the status check
            self.changed.emit(name)
        self.finished.emit(job_id, name, action, job.result, error)

    # Credentials ------------------------------------------------------------------------------
    def set_credential(self, key: str, value: str) -> str:
        """Store ``key`` in the OS keyring; returns "" or the reason it could not."""
        try:
            self.credentials.set(key, value.strip())
        except CredentialError as exc:
            return str(exc)
        self._credential_changed(key)
        return ""

    def remove_credential(self, key: str) -> str:
        try:
            self.credentials.delete(key)
        except CredentialError as exc:
            return str(exc)
        self._credential_changed(key)
        return ""

    def _credential_changed(self, key: str) -> None:
        for manifest in self.catalog.plugins.manifests():
            if any(spec.key == key for spec in manifest.credentials):
                self.changed.emit(manifest.name)
