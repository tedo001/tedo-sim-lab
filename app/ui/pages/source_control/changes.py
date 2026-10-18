"""Changes: what is staged and what is not, the diff of the selected file, stage / unstage /
discard, and the commit box (amend, push after committing)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.vcs import FileChange, RepoStatus

from ....services.source_control import SourceControl
from ...icons import icon
from ...widgets import DataTable, Pill, label
from ...widgets.diff_view import DiffView

__all__ = ["ChangesView"]

_TONE = {"M": ("Modified", "info"), "A": ("Added", "ok"), "D": ("Deleted", "fail"),
         "R": ("Renamed", "experimental"), "C": ("Copied", "experimental"), "T": ("Type", "info"),
         "?": ("New", "ok"), "U": ("Conflict", "fail")}


class ChangesView(QWidget):
    message = Signal(str)
    committed = Signal()

    def __init__(self, control: SourceControl, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.status: RepoStatus | None = None
        self._staged: list[FileChange] = []
        self._unstaged: list[FileChange] = []

        lists = QWidget()
        column = QVBoxLayout(lists)
        column.setContentsMargins(0, 0, 0, 0)
        self.staged_title = label("Staged", "CardTitle")
        column.addWidget(self.staged_title)
        self.staged_table = DataTable(("State", "File"), mono_columns=(1,), stretch_column=1)
        self.staged_table.currentCellChanged.connect(lambda row, *_: self._show(self._staged, row, True))
        column.addWidget(self.staged_table)
        row = QHBoxLayout()
        self.unstage_button = QPushButton(icon("arrow-down"), "Unstage")
        self.unstage_button.clicked.connect(self.unstage_selected)
        self.unstage_all_button = QPushButton("Unstage all")
        self.unstage_all_button.clicked.connect(lambda: self._unstage([c.path for c in self._staged]))
        row.addWidget(self.unstage_button)
        row.addWidget(self.unstage_all_button)
        row.addStretch(1)
        column.addLayout(row)
        self.changes_title = label("Changes", "CardTitle")
        column.addWidget(self.changes_title)
        self.changes_table = DataTable(("State", "File"), mono_columns=(1,), stretch_column=1)
        self.changes_table.currentCellChanged.connect(lambda row, *_: self._show(self._unstaged, row, False))
        column.addWidget(self.changes_table)
        row = QHBoxLayout()
        self.stage_button = QPushButton(icon("arrow-up"), "Stage")
        self.stage_button.clicked.connect(self.stage_selected)
        self.stage_all_button = QPushButton("Stage all")
        self.stage_all_button.clicked.connect(lambda: self._stage([c.path for c in self._unstaged]))
        self.discard_button = QPushButton(icon("rotate-ccw"), "Discard")
        self.discard_button.setToolTip("Throw away the selected file's unstaged changes (tracked files only)")
        self.discard_button.clicked.connect(lambda: self.discard_selected())
        for button in (self.stage_button, self.stage_all_button, self.discard_button):
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lists)
        scroll.setMinimumWidth(300)

        self.diff = DiffView()
        across = QSplitter(Qt.Orientation.Horizontal)
        across.addWidget(scroll)
        across.addWidget(self.diff)
        across.setStretchFactor(1, 1)
        across.setSizes([380, 620])

        self.message_box = QPlainTextEdit()
        self.message_box.setPlaceholderText("Commit message — the first line is the summary")
        self.message_box.setMaximumHeight(90)
        bottom = QHBoxLayout()
        self.amend = QCheckBox("Amend the last commit")
        self.push_after = QCheckBox("Push after committing")
        self.amend.toggled.connect(lambda _on: self._update())
        self.commit_button = QPushButton(icon("git-commit-horizontal"), "Commit")
        self.commit_button.setObjectName("Primary")
        self.commit_button.clicked.connect(self.commit)
        bottom.addWidget(self.amend)
        bottom.addWidget(self.push_after)
        bottom.addStretch(1)
        bottom.addWidget(self.commit_button)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(across, 1)
        box.addWidget(self.message_box)
        box.addLayout(bottom)
        self.show_status(None)

    # Showing ---------------------------------------------------------------------------------------
    def show_status(self, status: RepoStatus | None) -> None:
        self.status = status
        self._staged = status.staged if status else []
        self._unstaged = status.unstaged if status else []
        for table, changes, staged in ((self.staged_table, self._staged, True),
                                       (self.changes_table, self._unstaged, False)):
            table.clear_rows()
            for change in changes:
                code = change.index if staged else ("?" if change.untracked else change.worktree)
                text, tone = _TONE.get(code, (code, "planned"))
                name = f"{change.original} → {change.path}" if change.original and staged else change.path
                table.add_row([Pill(text, tone), name])
        self.staged_title.setText(f"Staged ({len(self._staged)})")
        self.changes_title.setText(f"Changes ({len(self._unstaged)})")
        self._update()
        if status is None:
            self.diff.show_diff("Open a repository to see its changes.")
        elif not self._staged and not self._unstaged:
            self.diff.show_diff("Nothing to commit: the working tree matches the last commit.")
        else:
            self.diff.show_diff("Choose a file to see what changed in it.")

    def _update(self) -> None:
        ready = self.status is not None
        self.commit_button.setEnabled(ready and (bool(self._staged) or self.amend.isChecked()))
        for button in (self.stage_button, self.stage_all_button, self.discard_button):
            button.setEnabled(ready and bool(self._unstaged))
        for button in (self.unstage_button, self.unstage_all_button):
            button.setEnabled(ready and bool(self._staged))

    def _show(self, changes: list[FileChange], row: int, staged: bool) -> None:
        if not 0 <= row < len(changes):
            return
        change = changes[row]
        git = self.control.git()
        self.control.run("Read a diff", lambda: git.diff(change, staged=staged), self.diff.show_diff,
                         self.message.emit)

    def _selected(self, table: DataTable, changes: list[FileChange]) -> list[str]:
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        return [changes[row].path for row in rows if row < len(changes)]

    # Acting ---------------------------------------------------------------------------------------------
    def _after(self, text: str):
        def done(_result) -> None:
            self.message.emit(text)
            self.control.changed.emit()
        return done

    def _stage(self, paths: list[str]) -> None:
        if paths:
            git = self.control.git()
            self.control.run("Stage files", lambda: git.stage(paths),
                             self._after(f"Staged {len(paths)} file(s)"), self.message.emit)

    def _unstage(self, paths: list[str]) -> None:
        if paths:
            git = self.control.git()
            self.control.run("Unstage files", lambda: git.unstage(paths),
                             self._after(f"Unstaged {len(paths)} file(s)"), self.message.emit)

    def stage_selected(self) -> None:
        self._stage(self._selected(self.changes_table, self._unstaged))

    def unstage_selected(self) -> None:
        self._unstage(self._selected(self.staged_table, self._staged))

    def discard_selected(self, *, confirm: bool = True) -> None:
        rows = sorted({index.row() for index in self.changes_table.selectionModel().selectedRows()})
        chosen = [self._unstaged[row] for row in rows if row < len(self._unstaged)]
        tracked = [c.path for c in chosen if not c.untracked]
        if not tracked:
            self.message.emit("Only changes to tracked files can be discarded (new files stay).")
            return
        if confirm and QMessageBox.question(
                self, "Discard changes", f"Throw away the unstaged changes to {', '.join(tracked)}? "
                "This cannot be undone.") != QMessageBox.StandardButton.Yes:
            return
        git = self.control.git()
        self.control.run("Discard changes", lambda: git.discard(tracked),
                         self._after(f"Discarded changes to {len(tracked)} file(s)"), self.message.emit)

    def commit(self) -> None:
        text = self.message_box.toPlainText()
        amend, push = self.amend.isChecked(), self.push_after.isChecked()
        git = self.control.git()

        def work() -> str:
            sha = git.commit(text, amend=amend)
            return f"{sha} pushed" if push and git.push() is not None else sha

        def done(result: str) -> None:
            self.message_box.clear()
            self.amend.setChecked(False)
            self.message.emit(f"Committed {result}")
            self.committed.emit()
            self.control.changed.emit()

        self.commit_button.setEnabled(False)
        self.control.run("Commit", work, done, lambda error: (self.message.emit(error), self._update()))
