"""Uncaught exceptions: log them (masked) and tell the person, never abort silently.

PyQt6 aborts the whole process on an exception raised inside a slot unless
``sys.excepthook`` is replaced, so this hook is installed before any window.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable
from types import TracebackType

from core.common import mask_text

__all__ = ["install_excepthook", "install_qt_message_handler"]

log = logging.getLogger("tedo.errors")

Hook = Callable[[type[BaseException], BaseException, TracebackType | None], None]


def install_excepthook(*, show_dialog: bool = True,
                       collect: list[str] | None = None) -> Hook:
    """Replace ``sys.excepthook``; return the previous hook.

    ``collect`` receives each masked traceback (the smoke test uses it to fail
    on errors raised inside slots).
    """
    previous = sys.excepthook
    showing = False

    def hook(kind: type[BaseException], error: BaseException,
             tb: TracebackType | None) -> None:
        nonlocal showing
        if issubclass(kind, KeyboardInterrupt):
            previous(kind, error, tb)
            return
        text = mask_text("".join(traceback.format_exception(kind, error, tb)))
        log.error("Uncaught exception\n%s", text)
        if collect is not None:
            collect.append(text)
        if show_dialog and not showing:
            showing = True
            try:
                from PyQt6.QtWidgets import QApplication, QMessageBox
                if QApplication.instance() is not None:
                    QMessageBox.critical(None, "Something went wrong",
                                         mask_text(f"{kind.__name__}: {error}") +
                                         "\n\nThe full traceback is in logs/app.log.")
            finally:
                showing = False

    sys.excepthook = hook
    return previous


#: Qt messages that are expected and say nothing about the lab (offscreen plugin noise).
_QUIET = ("propagateSizeHints",)


def install_qt_message_handler() -> None:
    """Send Qt's own warnings to the ``tedo.qt`` logger instead of stderr."""
    from PyQt6.QtCore import QtMsgType, qInstallMessageHandler

    qt_log = logging.getLogger("tedo.qt")
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING, QtMsgType.QtCriticalMsg: logging.ERROR,
              QtMsgType.QtFatalMsg: logging.CRITICAL}

    def handler(kind: QtMsgType, context: object, message: str) -> None:
        level = logging.DEBUG if any(q in message for q in _QUIET) else levels.get(kind, logging.INFO)
        qt_log.log(level, "%s", message)

    qInstallMessageHandler(handler)
