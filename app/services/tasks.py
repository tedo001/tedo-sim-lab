"""A ``QRunnable`` that runs ``fn(cancel_token, progress)`` and reports back by signal.

Signals are emitted from the pool thread; Qt queues them to the receiver's
thread (the UI), so slots never run on the worker thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from core.common.cancel import Cancelled, CancelToken, ProgressFn
from core.common.masking import mask_text

__all__ = ["TaskRunnable", "TaskSignals"]

log = logging.getLogger("tedo.tasks")


class TaskSignals(QObject):
    progress = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class TaskRunnable(QRunnable):
    def __init__(self, fn: Callable[[CancelToken, ProgressFn], Any], token: CancelToken) -> None:
        super().__init__()
        self.setAutoDelete(False)  # the queue holds it until finished; signals outlive run()
        self.fn = fn
        self.token = token
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.fn(self.token, lambda fraction, message="": self._emit(
                self.signals.progress, float(fraction), str(message)))
        except Cancelled:
            self._emit(self.signals.cancelled)
        except Exception as exc:
            log.exception("Background task failed")
            self._emit(self.signals.failed, mask_text(f"{type(exc).__name__}: {exc}"))
        else:
            if self.token.cancelled:
                self._emit(self.signals.cancelled)
            else:
                self._emit(self.signals.succeeded, result)

    @staticmethod
    def _emit(signal, *args) -> None:  # noqa: ANN001 (a bound signal)
        """A task that outlives the app's shutdown has no one left to tell."""
        try:
            signal.emit(*args)
        except RuntimeError:
            pass
