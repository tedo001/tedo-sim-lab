"""The terminal panel under every page (Ctrl+` or the top bar's terminal button): the same
terminals as the Terminal page, started in the Code page's project folder when one is open,
otherwise in the workspace."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from ...services.context import AppContext
from ..icons import icon
from ..pages.terminal_tabs import TerminalTabs
from ..theme.tokens import COLORS
from ..widgets import label

__all__ = ["TerminalPanel"]


class TerminalPanel(QFrame):
    close_requested = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("TerminalPanel")
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 6, 10, 8)
        box.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(label("Terminal", "CardTitle"))
        head.addStretch(1)
        close = QToolButton()
        close.setIcon(icon("x", COLORS["text_dim"]))
        close.setToolTip("Hide the terminal panel (Ctrl+`)")
        close.setAccessibleName("Hide the terminal panel")
        close.clicked.connect(self.close_requested.emit)
        head.addWidget(close)
        box.addLayout(head)
        self.terminals = TerminalTabs(ctx, folder=ctx.terminals.default_folder(), compact=True)
        self.terminals.open_path.connect(lambda path, line: ctx.open_file(path, line))
        box.addWidget(self.terminals, 1)
        ctx.terminals.project_changed.connect(self._project_changed)

    def _project_changed(self, root: Path | None) -> None:
        self.terminals.set_folder(root or self.ctx.paths.workspace)

    def open_if_empty(self) -> None:
        if not self.terminals.views():
            self.terminals.open_default()
        view = self.terminals.current_view()
        if view is not None:
            view.setFocus()

    def unsaved_work(self) -> list[str]:
        return self.terminals.unsaved_work()
