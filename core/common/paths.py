"""Where the lab keeps things.

The *code root* is the source checkout (this repository). The *workspace* is
where runtime data lives: datasets, checkpoints, run folders, the SQLite
database, MLflow's store and logs. By default they are the same folder, so a
fresh clone works with no setup; tests and the smoke test point the workspace
at a temporary directory instead.

Resolution order for the workspace: explicit argument, then the
``TEDO_LAB_WORKSPACE`` environment variable, then the code root when it is a
source checkout, otherwise ``~/TEDO-AI-Lab``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AppPaths", "CODE_ROOT", "WORKSPACE_DIRS", "WORKSPACE_ENV"]

#: The repository root: ``core/common/paths.py`` sits two levels below it.
CODE_ROOT = Path(__file__).resolve().parents[2]

WORKSPACE_ENV = "TEDO_LAB_WORKSPACE"

#: Folders a workspace always has. ``configs`` also holds the tracked card files
#: when the workspace is the code root.
WORKSPACE_DIRS = ("configs", "models", "datasets", "experiments", "notebooks",
                  "results", "reports", "database", "mlruns", "logs")


@dataclass(frozen=True)
class AppPaths:
    """Every path the application reads or writes, resolved once at startup."""

    code_root: Path
    workspace: Path

    @classmethod
    def resolve(cls, workspace: str | os.PathLike[str] | None = None, *,
                env: Mapping[str, str] | None = None,
                code_root: Path = CODE_ROOT) -> AppPaths:
        env = os.environ if env is None else env
        if workspace is not None:
            chosen = Path(workspace)
        elif env.get(WORKSPACE_ENV):
            chosen = Path(env[WORKSPACE_ENV])
        elif (code_root / "pyproject.toml").is_file():
            chosen = code_root
        else:
            chosen = Path.home() / "TEDO-AI-Lab"
        return cls(code_root=code_root, workspace=chosen.expanduser().resolve())

    def ensure(self) -> AppPaths:
        """Create the workspace folders that do not exist yet; return ``self``."""
        for name in WORKSPACE_DIRS:
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
        return self

    # Workspace folders --------------------------------------------------
    @property
    def configs(self) -> Path:
        return self.workspace / "configs"

    @property
    def models(self) -> Path:
        return self.workspace / "models"

    @property
    def datasets(self) -> Path:
        return self.workspace / "datasets"

    @property
    def experiments(self) -> Path:
        return self.workspace / "experiments"

    @property
    def notebooks(self) -> Path:
        return self.workspace / "notebooks"

    @property
    def results(self) -> Path:
        return self.workspace / "results"

    @property
    def reports(self) -> Path:
        return self.workspace / "reports"

    @property
    def database(self) -> Path:
        return self.workspace / "database"

    @property
    def mlruns(self) -> Path:
        return self.workspace / "mlruns"

    @property
    def logs(self) -> Path:
        return self.workspace / "logs"

    # Files ---------------------------------------------------------------
    @property
    def settings_file(self) -> Path:
        """The user's settings. Git-ignored; never holds credentials."""
        return self.configs / "settings.yaml"

    @property
    def lab_db(self) -> Path:
        return self.database / "lab.db"

    @property
    def mlflow_db(self) -> Path:
        return self.database / "mlflow.db"
