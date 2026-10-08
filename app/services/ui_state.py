"""Window layout remembered between sessions: sidebar width, split view, splitter sizes.

Stored with ``QSettings`` in the person's own settings (registry on Windows,
``~/.config`` on Linux), never in the install folder and never in a project:
the layout is a personal preference, not part of an experiment. Tests and the
smoke test pass a settings object backed by a file in a temporary folder.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings

__all__ = ["UiState"]


class UiState:
    def __init__(self, settings: QSettings | None = None) -> None:
        self.settings = settings if settings is not None else QSettings()

    @classmethod
    def in_file(cls, path: Path) -> UiState:
        return cls(QSettings(str(path), QSettings.Format.IniFormat))

    def _bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        return value in (True, "true", "1", 1) if not isinstance(value, bool) else value

    @property
    def sidebar_collapsed(self) -> bool:
        return self._bool("layout/sidebar_collapsed", False)

    @sidebar_collapsed.setter
    def sidebar_collapsed(self, value: bool) -> None:
        self.settings.setValue("layout/sidebar_collapsed", bool(value))

    @property
    def split_open(self) -> bool:
        return self._bool("layout/split_open", False)

    @split_open.setter
    def split_open(self, value: bool) -> None:
        self.settings.setValue("layout/split_open", bool(value))

    @property
    def split_page(self) -> str:
        return str(self.settings.value("layout/split_page", "hardware"))

    @split_page.setter
    def split_page(self, page_id: str) -> None:
        self.settings.setValue("layout/split_page", page_id)

    @property
    def splitter_state(self) -> QByteArray | None:
        value = self.settings.value("layout/splitter")
        return value if isinstance(value, QByteArray) else None

    @splitter_state.setter
    def splitter_state(self, state: QByteArray) -> None:
        self.settings.setValue("layout/splitter", state)

    def sync(self) -> None:
        self.settings.sync()
