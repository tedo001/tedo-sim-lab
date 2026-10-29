"""What every page is given: paths, settings (and a way to change them), credentials, the
catalogue, the lab's database, the job queue, hardware, experiments, logs, and a way to navigate.

Pages receive an :class:`AppContext` in their constructor instead of reaching
for globals, so a test can build any page against a temporary workspace.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.catalog import Catalog, load_catalog
from core.common import AppConfig, AppPaths, CredentialStore, experiment_python, packs
from core.common.paths import WORKSPACE_ENV
from core.experiment_engine.spec import load_spec
from core.tracking import LabStore
from core.tracking.mlflow_tracker import hold_no_mlflow_connections, release_mlflow_stores

from .benchmarks import BenchmarkService
from .colab import ColabService
from .downloads import DownloadService
from .experiment_setup import ExperimentSetup
from .experiments import ExperimentService
from .hardware import HardwareService
from .inference import InferenceService
from .jobs import JobQueue, worker_command
from .jupyter import JupyterLab, Notebooks
from .logs import LogService
from .mlflow_ui import MlflowUi
from .model_registry import ModelRegistryService
from .plugins import PluginService
from .settings import SettingsService
from .studies import StudyService
from .terminals import TerminalService
from .updates import UpdateCheck

__all__ = ["AppContext", "build_context"]


def _nowhere(page_id: str) -> None:  # until the main window installs the real one
    return None


def _no_editor(path, line: int = 1) -> None:  # until the main window installs the real one
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
    mlflow_ui: MlflowUi
    models: ModelRegistryService
    benchmarks: BenchmarkService
    colab: ColabService
    plugins: PluginService
    jupyter: JupyterLab
    notebooks: Notebooks
    setup: ExperimentSetup
    terminals: TerminalService
    settings: SettingsService
    logs: LogService
    updates: UpdateCheck
    inference: InferenceService
    studies: StudyService
    #: Switch the main window to another page; set by :class:`app.main_window.MainWindow`.
    navigate: Callable[[str], None] = field(default=_nowhere)
    #: Open a file at a line in the Code page; set by the main window.
    open_file: Callable[..., None] = field(default=_no_editor)
    #: Open a terminal in a folder (the Terminal page); set by the main window.
    open_terminal: Callable[..., None] = field(default=_no_editor)
    #: Switch to a page and return it (built on first use); set by the main window.
    show_page: Callable[[str], object] = field(default=_nowhere)

    def set_experiment_python(self, python: str) -> None:
        """Use ``python`` for workers, the hardware probe and plugin installs from now on, and
        remember it in this workspace's settings.yaml ("" = back to the default)."""
        self.settings.update(python_executable=python)
        current = experiment_python(self.config)
        self.catalog.plugins.set_python(current)
        self.hardware.set_python(current)
        self.inference.stop()  # the next request starts it under the new Python

    def close(self) -> None:
        """Stop background work and close the database (app exit)."""
        self.terminals.shutdown()
        self.inference.stop()
        self.logs.close()
        self.hardware.stop()
        self.mlflow_ui.stop()
        self.jupyter.stop()
        self.jobs.shutdown()
        self.store.close()
        release_mlflow_stores()


def _python_for_run(run_dir: Path, config: AppConfig, catalog: Catalog) -> str:
    """The experiment Python, or an installed environment pack that serves the run's runner."""
    base = experiment_python(config)
    try:
        runner = catalog.runners.for_spec(load_spec(Path(run_dir) / "experiment.yaml")).id
    except Exception:  # a benchmark or evaluation folder, or an unreadable spec: the core Python
        return base
    return packs.python_for(base, runner=runner) if base else base


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
    hold_no_mlflow_connections()  # the window can always let go of mlflow.db
    python = experiment_python(config)
    catalog = load_catalog(paths, credentials, python=python)
    store = LabStore.open(paths)
    store.recover_interrupted()
    # The experiment Python is read at each start, so setting it up takes effect at once.
    jobs = JobQueue(lambda run_dir: worker_command(run_dir,
                                                   python=_python_for_run(run_dir, config, catalog),
                                                   code_root=paths.code_root, workspace=paths.workspace),
                    max_concurrent_runs=config.max_concurrent_runs, store=store)
    hardware = HardwareService(python=python, code_root=paths.code_root, probe=probe_hardware)
    experiments = ExperimentService(paths, store, jobs, catalog)
    experiments.python = lambda: experiment_python(config)
    downloads = DownloadService(catalog.datasets, jobs)
    models = ModelRegistryService(paths, config, store, experiments, jobs)
    terminals = TerminalService(paths, config)
    return AppContext(paths, config, credentials, catalog, store, jobs, hardware, experiments, downloads,
                      MlflowUi(paths, config), models, BenchmarkService(paths, jobs),
                      ColabService(paths, experiments, jobs),
                      PluginService(paths, catalog, credentials, jobs), JupyterLab(paths, config),
                      Notebooks(paths, store), ExperimentSetup(jobs), terminals,
                      SettingsService(paths, config, jobs, terminals), LogService(paths), UpdateCheck(jobs),
                      InferenceService(paths, config), StudyService(paths, store, experiments))
