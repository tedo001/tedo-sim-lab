"""History: the latest commits on the current branch; the selected one's message and diff."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QScrollArea, QSplitter, QVBoxLayout, QWidget

from core.vcs import Commit

from ....services.source_control import SourceControl
from ...widgets import DataTable, label
from ...widgets.diff_view import DiffView

__all__ = ["HistoryView"]

LIMIT = 200


class HistoryView(QWidget):
    message = Signal(str)

    def __init__(self, control: SourceControl, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.commits: list[Commit] = []
        self.caption = label("", "CardCaption")
        self.table = DataTable(("Commit", "Summary", "Author", "Date", "Refs"), mono_columns=(0, 3),
                               stretch_column=1)
        self.table.cellClicked.connect(lambda row, _c: self.show_commit(row))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.table)
        column.addStretch(1)
        scroll.setWidget(holder)
        self.diff = DiffView()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(scroll)
        split.addWidget(self.diff)
        split.setSizes([300, 300])
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.caption)
        box.addWidget(split, 1)

    def refresh(self) -> None:
        git = self.control.git()
        self.caption.setText("Reading the history…")
        self.control.run("Read the history", lambda: git.log(LIMIT), self.show_commits, self.message.emit)

    def show_commits(self, commits: list[Commit]) -> None:
        self.commits = commits
        self.table.clear_rows()
        for commit in commits:
            refs = ", ".join(part.strip() for part in commit.refs.split(",") if part.strip())
            self.table.add_row([commit.short, commit.subject, commit.author,
                                commit.date[:16].replace("T", " "), refs])
        count = len(commits)
        self.caption.setText("No commits yet." if not count else
                             f"The latest {count} commit{'s' if count != 1 else ''}"
                             + (f" (the newest {LIMIT})" if count == LIMIT else ""))

    def show_commit(self, row: int) -> None:
        if 0 <= row < len(self.commits):
            git, sha = self.control.git(), self.commits[row].sha
            self.control.run("Read a commit", lambda: git.show(sha), self.diff.show_diff, self.message.emit)
