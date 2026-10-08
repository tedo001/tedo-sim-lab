"""Optional dependencies: find out whether they are installed without importing them.

Availability is decided by *distribution metadata*, not by ``find_spec``. The
workspace has folders called ``datasets/`` and ``models/``; with the repo root
on ``sys.path`` they would look like importable namespace packages and fool a
``find_spec("datasets")`` check into reporting Hugging Face Datasets as present.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from importlib import metadata
from types import ModuleType

__all__ = ["distribution_version", "is_installed", "optional_import"]

log = logging.getLogger("tedo.optional")


def _names(distribution: str | Sequence[str]) -> Sequence[str]:
    return (distribution,) if isinstance(distribution, str) else distribution


def distribution_version(distribution: str | Sequence[str]) -> str | None:
    """Installed version of the first of ``distribution`` found, else ``None``."""
    for name in _names(distribution):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def is_installed(distribution: str | Sequence[str]) -> bool:
    return distribution_version(distribution) is not None


def optional_import(module: str, distribution: str | Sequence[str] | None = None
                    ) -> ModuleType | None:
    """Import ``module`` if its distribution is installed, else return ``None``.

    ``distribution`` defaults to the module's top-level name; pass it when they
    differ, e.g. ``optional_import("cv2", ("opencv-python-headless", "opencv-python"))``.
    A distribution that is installed but fails to import is logged and treated
    as missing, so a broken optional package never stops the app starting.
    """
    if not is_installed(distribution or module.split(".")[0]):
        return None
    try:
        return importlib.import_module(module)
    except Exception as exc:
        log.warning("%s is installed but failed to import: %s", module, exc)
        return None
