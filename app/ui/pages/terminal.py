"""Terminal: real terminals in tabs — bash, zsh, PowerShell, Command Prompt, Git Bash, WSL, the
experiment Python, vim, Neovim, Docker containers, or a command of your own. Full-screen programs
work (vim, nvim, top, htop, less); the experiment Python comes first on PATH and the workspace is
in ``TEDO_LAB_WORKSPACE``. The same terminals open on any page with Ctrl+`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

from ...services.context import AppContext
from ..widgets import Page, label
from .terminal_tabs import TerminalTabs

__all__ = ["TerminalPage"]


class TerminalPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Terminal", "shells, Python, vim, Neovim and Docker in real terminals", parent)
        self.ctx = ctx
        self.terminals = TerminalTabs(ctx, folder=ctx.paths.workspace)
        self.terminals.setMinimumHeight(460)
        self.terminals.open_path.connect(self._open_path)
        self.body.addWidget(self.terminals, 1)
        self.body.addWidget(label(
            "Copy and paste with Ctrl+Shift+C and Ctrl+Shift+V (Ctrl+C copies while text is selected). "
            "Ctrl+click opens a link or a file. Shift+PgUp/PgDn scrolls back. Ctrl+` opens a terminal "
            "panel on any page. The experiment Python comes first on PATH.", "CardCaption", wrap=True))
        self._opened = False

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        if not self._opened:  # the first visit opens the default shell
            self._opened = True
            QTimer.singleShot(0, self._open_first)

    def _open_first(self) -> None:
        if not self.terminals.views():
            self.terminals.open_default()

    def _open_path(self, path: Path, line: int) -> None:
        self.ctx.open_file(path, line)

    def unsaved_work(self) -> list[str]:
        return self.terminals.unsaved_work()
