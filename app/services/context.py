"""What every page is given: paths, settings, credentials, the catalogue, the lab's
database, the job queue, hardware, experiments, and a way to navigate.

Pages receive an :class:`AppContext` in their constructor instead of reaching
for globals, so a test can build any page against a temporary workspace.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from core.catalog import Catalog, load_catalog
from core.common import AppConfig, AppPaths, CredentialStore, experiment_python
from core.common.paths import WORKSPACE_ENV
from core.tracking import LabStore

from .downloads import DownloadService
from .experiments import ExperimentService
from .hardware import HardwareService
from .jobs import JobQueue, worker_command

__all__ = ["AppContext", "build_context"]


def _nowhere(page_id: str) -> None:  # until the main window installs the real one
    return None


@dataclass
class AppContext:
    paths: AppPaths
    config: AppConfig
    credentials: CredentialStore
    catalog: Catalog
    store: LabStore
    jobs: JobQueue
    hardware: HardwareService
    experiments: ExperimentService
    downloads: DownloadService
    #: Switch the main window to another page; set by :class:`app.main_window.MainWindow`.
    navigate: Callable[[str], None] = field(default=_nowhere)

    def close(self) -> None:
        """Stop background work and close the database (app exit)."""
        self.hardware.stop()
        self.jobs.shutdown()
        self.store.close()


def build_context(paths: AppPaths, config: AppConfig, credentials: CredentialStore | None = None,
                  *, probe_hardware: bool = True) -> AppContext:
    """Open the workspace: load the catalogue, open (and migrate) the database, start the queue.

    Anything left queued or running by a previous session is marked interrupted.
    Needs a ``QApplication`` (the job queue is a ``QObject``). ``probe_hardware=False``
    skips the PyTorch/CUDA probe process (tests); sampling still works.
    """
    credentials = credentials or CredentialStore()
    # Lab code that runs in this process (runner checks, dataset status) resolves the workspace
    # the same way a worker does.
    os.environ[WORKSPACE_ENV] = str(paths.workspace)
    python = experiment_python(config)
    catalog = load_catalog(paths, credentials, python=python)
    store = LabStore.open(paths)
    store.recover_interrupted()
    jobs = JobQueue(lambda run_dir: worker_command(run_dir, python=python, code_root=paths.code_root,
                                                   workspace=paths.workspace),
                    max_concurrent_runs=config.max_concurrent_runs, store=store)
    hardware = HardwareService(python=python, code_root=paths.code_root, probe=probe_hardware)
    experiments = ExperimentService(paths, store, jobs, catalog)
    downloads = DownloadService(catalog.datasets, jobs)
    return AppContext(paths, config, credentials, catalog, store, jobs, hardware, experiments, downloads)
