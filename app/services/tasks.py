"""A ``QRunnable`` that runs ``fn(cancel_token, progress)`` and reports back by signal.

Signals are emitted from the pool thread; Qt queues them to the receiver's
thread (the UI), so slots never run on the worker thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.common.cancel import Cancelled, CancelToken, ProgressFn
from core.common.masking import mask_text

__all__ = ["TaskRunnable", "TaskSignals"]

log = logging.getLogger("tedo.tasks")


class TaskSignals(QObject):
    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class TaskRunnable(QRunnable):
    def __init__(self, fn: Callable[[CancelToken, ProgressFn], Any], token: CancelToken) -> None:
        super().__init__()
        self.setAutoDelete(False)  # the queue holds it until finished; signals outlive run()
        self.fn = fn
        self.token = token
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.fn(self.token, lambda fraction, message="": self.signals.progress.emit(
                float(fraction), str(message)))
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as exc:
            log.exception("Background task failed")
            self.signals.failed.emit(mask_text(f"{type(exc).__name__}: {exc}"))
        else:
            if self.token.cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.succeeded.emit(result)
