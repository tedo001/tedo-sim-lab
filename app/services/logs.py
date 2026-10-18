"""Logs in the window: the lab's log files (``app.log``, ``errors.log``, older rotations, each
run's ``run.log``) and a count of the problems logged since the app started, for the top bar.

Records can come from any thread; the count reaches the window through a queued signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal

from core.common import AppPaths
from core.common.logging_setup import APP_LOG, ERRORS_LOG
from core.common.masking import mask_text

__all__ = ["LogService"]


class _Counter(logging.Handler):
    def __init__(self, service: LogService) -> None:
        super().__init__(logging.ERROR)
        self.service = service

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.service._recorded.emit(mask_text(f"{record.name}: {record.getMessage()}"))
        except RuntimeError:  # the service is gone (a late record during shutdown)
            pass


class LogService(QObject):
    #: The number of errors logged since start (or since :meth:`clear_problems`), and the latest.
    problems_changed = Signal(int, str)
    _recorded = Signal(str)

    def __init__(self, paths: AppPaths, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.problems = 0
        self.latest = ""
        self._recorded.connect(self._count, Qt.ConnectionType.QueuedConnection)
        self._handler = _Counter(self)
        logging.getLogger().addHandler(self._handler)

    @property
    def app_log(self) -> Path:
        return self.paths.logs / APP_LOG

    @property
    def errors_log(self) -> Path:
        return self.paths.logs / ERRORS_LOG

    def files(self) -> list[Path]:
        """The log folder's files, newest first (app.log, errors.log and their rotations)."""
        if not self.paths.logs.is_dir():
            return []
        found = [p for p in self.paths.logs.iterdir() if p.is_file() and ".log" in p.name]
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def _count(self, text: str) -> None:
        self.problems += 1
        self.latest = text
        self.problems_changed.emit(self.problems, text)

    def clear_problems(self) -> None:
        self.problems, self.latest = 0, ""
        self.problems_changed.emit(0, "")

    def close(self) -> None:
        logging.getLogger().removeHandler(self._handler)
