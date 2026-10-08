"""User settings: ``<workspace>/configs/settings.yaml``.

The file is optional and git-ignored; anything missing takes the default
below. It must never hold credentials: a key that looks like one is rejected
with a message pointing at the keyring or an environment variable, and so is
any key the lab does not know.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .paths import AppPaths

__all__ = ["AppConfig", "ConfigError", "experiment_python", "load_config", "mlflow_tracking_uri",
           "save_config"]

_CREDENTIAL_KEY = re.compile(r"(?i)(key|token|secret|password|passwd|credential)")


class ConfigError(ValueError):
    """The settings file could not be used; the message says why."""


class AppConfig(BaseModel):
    """Settings a person may change. Defaults suit a single CPU or one GPU."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: Mirror every run to MLflow (when it is installed in the experiment Python).
    mlflow_tracking: bool = True
    #: Empty = SQLite in ``<workspace>/database/mlflow.db``.
    mlflow_tracking_uri: str = ""
    #: ``auto`` = PowerShell on Windows, ``$SHELL`` or bash elsewhere.
    terminal_shell: str = "auto"
    max_concurrent_runs: int = Field(default=1, ge=1, le=8)
    default_device: str = "auto"
    #: Python that runs experiments and installs plugin packages. Empty = the app's own
    #: interpreter. The packaged (installer) app needs a real Python environment here.
    python_executable: str = ""


def experiment_python(config: AppConfig) -> str:
    """The interpreter experiment workers run under."""
    return config.python_executable or sys.executable


def _reject_credentials(data: dict[str, Any]) -> None:
    for key in data:
        if _CREDENTIAL_KEY.search(str(key)):
            raise ConfigError(
                f"settings.yaml must not contain credentials ({key!r}). Set an "
                "environment variable or store it in the OS keyring from the Plugin Store.")


def load_config(paths: AppPaths) -> AppConfig:
    """Read the settings file, or return the defaults when there is none."""
    file = paths.settings_file
    if not file.is_file():
        return AppConfig()
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{file} must be a mapping of setting: value")
    _reject_credentials(data)
    try:
        return AppConfig(**data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{file}: {problems}") from None


def save_config(config: AppConfig, paths: AppPaths) -> Path:
    """Write only the settings that differ from the defaults; return the file."""
    changed = config.model_dump(exclude_defaults=True)
    paths.settings_file.parent.mkdir(parents=True, exist_ok=True)
    paths.settings_file.write_text(yaml.safe_dump(changed, sort_keys=True), encoding="utf-8")
    return paths.settings_file


def mlflow_tracking_uri(config: AppConfig, paths: AppPaths) -> str:
    """The configured MLflow URI, or the workspace's SQLite store."""
    return config.mlflow_tracking_uri or f"sqlite:///{paths.mlflow_db.as_posix()}"
