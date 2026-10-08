"""Cooperative cancellation shared by runners, downloads and background tasks."""

from __future__ import annotations

import threading
from collections.abc import Callable

__all__ = ["CancelToken", "Cancelled", "ProgressFn"]

#: ``progress(fraction, message)``: fraction in 0..1, or a negative number when unknown.
ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    """Raised by :meth:`CancelToken.raise_if_cancelled` once cancellation was requested."""


class CancelToken:
    """Set from one thread (the UI, a stdin watcher), checked from another (the work)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled("cancelled on request")

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds, waking early on cancel; return ``cancelled``."""
        return self._event.wait(timeout)
