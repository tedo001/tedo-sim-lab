"""Application logging: rotating files in ``<workspace>/logs`` plus stderr.

``app.log`` holds everything at the configured level; ``errors.log`` keeps only warnings and
errors, so problems stay findable long after app.log has rotated. Handlers sit on the root
logger so records from third-party libraries (MLflow, urllib3, ...) pass through the same
:class:`SecretMaskingFilter`. The root stays at WARNING; the lab's own ``tedo.*`` loggers use
the configured level (:func:`set_level` changes it while the app runs).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .masking import SecretMaskingFilter

__all__ = ["APP_LOG", "ERRORS_LOG", "LOG_FORMAT", "get_logger", "set_level", "setup_logging"]

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
APP_LOG, ERRORS_LOG = "app.log", "errors.log"
_OWNED = "_tedo_handler"


def get_logger(name: str) -> logging.Logger:
    """A logger under the ``tedo`` namespace: ``get_logger("ui")`` → ``tedo.ui``."""
    return logging.getLogger(name if name.startswith("tedo") else f"tedo.{name}")


def setup_logging(log_dir: Path | None, level: str = "INFO", *,
                  console: bool = True) -> logging.Logger:
    """Install the lab's handlers (idempotent) and return the ``tedo`` logger."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _OWNED, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_dir / APP_LOG, maxBytes=5_000_000,
                                            backupCount=3, encoding="utf-8"))
        problems = RotatingFileHandler(log_dir / ERRORS_LOG, maxBytes=2_000_000, backupCount=2,
                                       encoding="utf-8")
        problems.setLevel(logging.WARNING)
        handlers.append(problems)
    if console and sys.stderr is not None:  # a windowed .exe has no console streams at all
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(SecretMaskingFilter())
        setattr(handler, _OWNED, True)
        root.addHandler(handler)

    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    return set_level(level)


def set_level(level: str) -> logging.Logger:
    """The lab's loggers record ``level`` and above from now on; returns the ``tedo`` logger."""
    lab = logging.getLogger("tedo")
    lab.setLevel(level.upper())
    return lab
