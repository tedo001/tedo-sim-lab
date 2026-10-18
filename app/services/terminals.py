"""Every terminal the app has open, wherever it is shown (Terminal page, bottom panel, Code page):
the profiles this machine can run, the environment terminals get, starting sessions, and ending
all of them when the app closes (so no shell outlives the window)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.common import AppConfig, AppPaths, experiment_python
from core.terminal import Profile, PtyError, profiles, terminal_environment

from .pty_session import PtySession

__all__ = ["PtyError", "TerminalService"]


class TerminalService(QObject):
    #: The Code page's project folder changed (the bottom panel starts terminals there).
    project_changed = Signal(object)

    def __init__(self, paths: AppPaths, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config = paths, config
        self.sessions: list[PtySession] = []
        self.project_root: Path | None = None
        self._profiles: list[Profile] | None = None

    def profiles(self, *, refresh: bool = False) -> list[Profile]:
        if self._profiles is None or refresh:
            self._profiles = profiles(self.config.terminal_shell, experiment_python(self.config))
        return self._profiles

    def default_folder(self) -> Path:
        return self.project_root or self.paths.workspace

    def set_project(self, root: Path | None) -> None:
        self.project_root = root
        self.project_changed.emit(root)

    def environment(self) -> dict[str, str]:
        return terminal_environment(self.paths.workspace, experiment_python(self.config))

    def start(self, argv: list[str], cwd: Path, *, title: str = "", rows: int = 24,
              cols: int = 80) -> PtySession:
        """A running session; ``PtyError`` when the program cannot start."""
        session = PtySession(argv, cwd, self.environment(), rows=rows, cols=cols, title=title)
        session.start()
        self.sessions.append(session)
        return session

    def end(self, session: PtySession) -> None:
        session.terminate()
        if session in self.sessions:
            self.sessions.remove(session)
        session.deleteLater()

    def shutdown(self) -> None:
        for session in list(self.sessions):
            session.terminate()
        self.sessions.clear()
