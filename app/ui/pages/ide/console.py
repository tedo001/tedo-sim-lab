"""The Code page's bottom panel: Run output (double-click a traceback line to open that file at
that line), real terminals in the project folder (the debugger runs in one), and the project's
git state."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QMouseEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.context import AppContext
from ....services.project import TRACEBACK_LINE, git_status
from ...icons import icon
from ...theme import mono_font
from ...theme.tokens import COLORS
from ...widgets import DataTable, label
from ..terminal_tabs import TerminalTabs

__all__ = ["BottomPanel", "OutputView"]

MAX_LINES = 10_000


class OutputView(QPlainTextEdit):
    """Read-only output; double-clicking a ``File "…", line N`` line asks to open it."""

    jump = Signal(Path, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Code")
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setMaximumBlockCount(MAX_LINES)

    def write(self, text: str, stream: str = "stdout") -> None:
        colour = {"stderr": COLORS["fail"], "info": COLORS["text_faint"]}.get(stream, COLORS["text"])
        style = QTextCharFormat()
        style.setForeground(QColor(colour))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, style)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        line = self.cursorForPosition(event.position().toPoint()).block().text()
        if self.open_line(line):
            return
        super().mouseDoubleClickEvent(event)

    def open_line(self, line: str) -> bool:
        match = TRACEBACK_LINE.search(line)
        if match and Path(match["path"]).is_file():
            self.jump.emit(Path(match["path"]), int(match["line"]))
            return True
        return False


class BottomPanel(QTabWidget):
    jump = Signal(Path, int)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setDocumentMode(True)
        self.output = OutputView()
        self.output.jump.connect(self.jump)
        self.addTab(self.output, icon("play"), "Run")

        self.terminals = TerminalTabs(ctx, folder=ctx.paths.workspace, compact=True)
        self.terminals.open_path.connect(self.jump)
        self.terminal_index = self.addTab(self.terminals, icon("terminal"), "Terminal")
        self.currentChanged.connect(self._tab_changed)

        git = QWidget()
        box = QVBoxLayout(git)
        box.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.branch = label("", "Mono")
        refresh = QPushButton(icon("refresh-cw"), "Refresh")
        refresh.clicked.connect(self.refresh_git)
        full = QPushButton(icon("git-branch"), "Git & GitHub")
        full.setToolTip("Stage, commit, branches, history, push and GitHub")
        full.clicked.connect(lambda: ctx.navigate("source_control"))
        row.addWidget(self.branch, 1)
        row.addWidget(refresh)
        row.addWidget(full)
        box.addLayout(row)
        self.changes = DataTable(("Change", "File"), mono_columns=(0, 1), stretch_column=1)
        self.changes.cellDoubleClicked.connect(self._open_change)
        box.addWidget(self.changes)
        box.addStretch(1)
        self.addTab(git, icon("git-branch"), "Git")
        self.root: Path | None = None
        self._git_job: str | None = None
        self._changed_paths: list[Path] = []
        ctx.jobs.job_finished.connect(self._git_done)

    def set_root(self, root: Path) -> None:
        self.root = root
        self.terminals.set_folder(root)
        self.refresh_git()

    def _tab_changed(self, index: int) -> None:
        if index == self.terminal_index and not self.terminals.views():
            self.terminals.open_default()  # the first visit opens a shell in the project

    def show_terminal(self) -> None:
        self.setCurrentIndex(self.terminal_index)

    # Git ---------------------------------------------------------------------------------------
    def refresh_git(self) -> None:
        if self.root is None or self._git_job is not None:
            return
        root = self.root
        self.branch.setText("Reading git…")
        self._git_job = self.ctx.jobs.submit_task(lambda cancel, progress: (root, git_status(root)),
                                                  title="Read git status", quiet=True)

    def _git_done(self, job_id: str, status: str) -> None:
        if job_id != self._git_job:
            return
        self._git_job = None
        self.changes.clear_rows()
        self._changed_paths = []
        if status != "completed":
            self.branch.setText("git status could not be read")
            return
        root, state = self.ctx.jobs.job(job_id).result
        if state.error:
            self.branch.setText(f"Git: {state.error}")
            return
        count = len(state.changes)
        self.branch.setText(f"Branch {state.branch or '(none)'} · {count or 'no'} changed "
                            f"file{'' if count == 1 else 's'}")
        top = _repository_root(root)
        for code, name in state.changes:
            self.changes.add_row([code, name])
            self._changed_paths.append((top or root) / name.split(" -> ")[-1].strip('"'))

    def _open_change(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._changed_paths) and self._changed_paths[row].is_file():
            self.jump.emit(self._changed_paths[row], 1)


def _repository_root(folder: Path) -> Path | None:
    for candidate in (folder, *folder.parents):
        if (candidate / ".git").exists():
            return candidate
    return None
