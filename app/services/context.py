"""What every page is given: paths, settings, credentials and a way to navigate.

Pages receive an :class:`AppContext` in their constructor instead of reaching
for globals, so a test can build any page against a temporary workspace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.common import AppConfig, AppPaths, CredentialStore

__all__ = ["AppContext"]


def _nowhere(page_id: str) -> None:  # until the main window installs the real one
    return None


@dataclass
class AppContext:
    paths: AppPaths
    config: AppConfig
    credentials: CredentialStore
    #: Switch the main window to another page; set by :class:`app.main_window.MainWindow`.
    navigate: Callable[[str], None] = field(default=_nowhere)
