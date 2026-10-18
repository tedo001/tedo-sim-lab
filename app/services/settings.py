"""Changing settings while the app runs: check the new values together, save settings.yaml, and
apply what can take effect at once. Every service holds the same :class:`AppConfig` object, so
they all see a change as soon as it is made."""

from __future__ import annotations

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from core.common import AppConfig, AppPaths, save_config
from core.common.config import ConfigError
from core.common.logging_setup import set_level

from .jobs import JobQueue
from .terminals import TerminalService

__all__ = ["LATER", "SettingsService"]

#: Settings that take effect only for what starts later, and when.
LATER = {
    "mlflow_tracking": "applies to runs started from now on",
    "mlflow_tracking_uri": "applies to new runs and the next start of the MLflow UI",
    "default_device": "is what the builder starts experiments on, unless one was chosen there",
}


class SettingsService(QObject):
    #: The names of the settings that changed.
    changed = Signal(object)

    def __init__(self, paths: AppPaths, config: AppConfig, jobs: JobQueue, terminals: TerminalService,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config, self.jobs, self.terminals = paths, config, jobs, terminals

    def update(self, **changes) -> set[str]:  # noqa: ANN003 (setting values)
        """Apply ``changes`` and save them; ``ConfigError`` says what is wrong (nothing changes
        then). Returns the names of the settings whose value changed."""
        try:
            checked = AppConfig(**{**self.config.model_dump(), **changes})
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ConfigError(problems) from None
        changed = {name for name in changes if getattr(self.config, name) != getattr(checked, name)}
        for name in changed:
            setattr(self.config, name, getattr(checked, name))
        if changed:
            save_config(self.config, self.paths)
            self._apply(changed)
            self.changed.emit(changed)
        return changed

    def _apply(self, changed: set[str]) -> None:
        if "log_level" in changed:
            set_level(self.config.log_level)
        if "max_concurrent_runs" in changed:
            self.jobs.set_max_concurrent_runs(self.config.max_concurrent_runs)
        if changed & {"terminal_shell", "python_executable"}:
            self.terminals.profiles(refresh=True)
