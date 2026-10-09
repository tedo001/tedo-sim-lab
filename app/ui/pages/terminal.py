"""Terminal: run shell commands in the workspace (PowerShell on Windows, bash elsewhere), one at
a time, with history (Up/Down), cd, Stop and the experiment Python first on PATH. Not a
terminal emulator: commands cannot be typed into, and full-screen programs do not work."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QWidget

from ...services.context import AppContext
from ...services.terminal import ShellSession, shell_command
from ..icons import icon
from ..theme import mono_font
from ..theme.tokens import COLORS
from ..widgets import Card, Page, PathLabel, label

__all__ = ["TerminalPage"]

#: Lines kept on screen; older output scrolls away.
MAX_LINES = 5_000


class TerminalPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Terminal", "commands in the workspace, one at a time", parent)
        self.ctx = ctx
        self.session = ShellSession(ctx.paths, ctx.config, self)
        self._history_index = 0
        program, _ = shell_command(ctx.config, "")
        card = Card("Shell", f"{program} · the experiment Python comes first on PATH")
        where = QHBoxLayout()
        where.addWidget(label("Folder", "KvKey"))
        self.folder = PathLabel(self.session.cwd)
        where.addWidget(self.folder, 1)
        self.state = label("Ready", "CardCaption")
        where.addWidget(self.state)
        card.add(where)
        self.screen = QPlainTextEdit()
        self.screen.setReadOnly(True)
        self.screen.setFont(mono_font())
        self.screen.setMaximumBlockCount(MAX_LINES)
        self.screen.setMinimumHeight(360)
        self.screen.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        card.add(self.screen)
        row = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setFont(mono_font())
        self.prompt.setPlaceholderText("Type a command and press Enter (Up/Down for history)")
        self.prompt.returnPressed.connect(self.run)
        self.prompt.installEventFilter(self)
        row.addWidget(self.prompt, 1)
        self.run_button = QPushButton(icon("play"), "Run")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton(icon("square"), "Stop")
        self.stop_button.clicked.connect(self.session.stop)
        self.clear_button = QPushButton(icon("eraser"), "Clear")
        self.clear_button.clicked.connect(self.screen.clear)
        for button in (self.run_button, self.stop_button, self.clear_button):
            row.addWidget(button)
        card.add(row)
        card.add(label("Commands cannot be typed into once running (their input is closed), and "
                       "full-screen programs such as vim or top do not work here. Long jobs belong in "
                       "the Experiment Builder, where they are tracked.", "CardCaption", wrap=True))
        self.body.addWidget(card)
        self.body.addStretch(1)
        self.session.output.connect(self._write)
        self.session.started.connect(self._started)
        self.session.finished.connect(self._finished)
        self._update()

    # Running ------------------------------------------------------------------------------------
    def run(self, command: str | None = None) -> bool:
        text = self.prompt.text() if command is None or isinstance(command, bool) else command
        if not text.strip() or self.session.running:
            return False
        if text.strip() in ("clear", "cls"):
            self.screen.clear()
            self.prompt.clear()
            return True
        self.prompt.clear()
        return self.session.run(text)

    def _started(self, command: str) -> None:
        self._history_index = len(self.session.history)
        self._write(f"{self.session.cwd}> {command}\n", "info")
        self._update()

    def _finished(self, code: int, stopped: bool) -> None:
        self.folder.set_full_text(str(self.session.cwd))
        if stopped:
            self._write("[stopped]\n", "info")
        elif code != 0:
            self._write(f"[exit code {code}]\n", "info")
        self._update()

    def _write(self, text: str, stream: str) -> None:
        colour = {"stderr": COLORS["fail"], "info": COLORS["text_faint"]}.get(stream, COLORS["text"])
        style = QTextCharFormat()
        style.setForeground(QColor(colour))
        cursor = self.screen.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, style)
        self.screen.setTextCursor(cursor)
        self.screen.ensureCursorVisible()

    def text(self) -> str:
        return self.screen.toPlainText()

    def _update(self) -> None:
        running = self.session.running
        self.state.setText("Running…" if running else "Ready")
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    # History ------------------------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.prompt and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            history = self.session.history
            if key == Qt.Key.Key_Up and history:
                self._history_index = max(0, self._history_index - 1)
                self.prompt.setText(history[self._history_index])
                return True
            if key == Qt.Key.Key_Down and history:
                self._history_index = min(len(history), self._history_index + 1)
                index = self._history_index
                self.prompt.setText(history[index] if index < len(history) else "")
                return True
        return super().eventFilter(watched, event)
