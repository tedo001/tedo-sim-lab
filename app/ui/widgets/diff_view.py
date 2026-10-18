"""A read-only view of a unified diff: added lines green, removed lines red, hunk headers in the
accent colour, file headers faint."""

from __future__ import annotations

from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ..theme import mono_font
from ..theme.tokens import COLORS

__all__ = ["DiffView"]


def _style(colour: str, background: str | None = None) -> QTextCharFormat:
    style = QTextCharFormat()
    style.setForeground(QColor(colour))
    if background:
        tint = QColor(background)
        tint.setAlpha(28)
        style.setBackground(tint)
    return style


class DiffHighlighter(QSyntaxHighlighter):
    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self.added = _style(COLORS["ok"], COLORS["ok"])
        self.removed = _style(COLORS["fail"], COLORS["fail"])
        self.hunk = _style(COLORS["info"])
        self.header = _style(COLORS["text_faint"])

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt API)
        if text.startswith(("+++", "---", "diff --git", "index ", "new file", "deleted file",
                            "similarity", "rename ", "commit ", "Author", "AuthorDate", "Commit",
                            "CommitDate", "Merge:")):
            style = self.header
        elif text.startswith("@@"):
            style = self.hunk
        elif text.startswith("+"):
            style = self.added
        elif text.startswith("-"):
            style = self.removed
        else:
            return
        self.setFormat(0, len(text), style)


class DiffView(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Code")
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setPlaceholderText("Select a file or a commit to see its changes.")
        self.highlighter = DiffHighlighter(self.document())

    def show_diff(self, text: str) -> None:
        self.setPlainText(text if text.strip() else "(no changes to show)")
