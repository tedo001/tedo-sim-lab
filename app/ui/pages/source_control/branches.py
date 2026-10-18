"""Branches & remotes: switch or make branches, add remotes, and the name and email git puts
on commits (for this repository or all of them)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.vcs import Branch

from ....services.source_control import SourceControl
from ...icons import icon
from ...widgets import Card, DataTable, Pill

__all__ = ["BranchesView"]


class BranchesView(QWidget):
    message = Signal(str)

    def __init__(self, control: SourceControl, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.branch_list: list[Branch] = []
        self.remote_list: list[tuple[str, str]] = []
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        branches = Card("Branches", "local and remote")
        self.branches = DataTable(("Branch", "Kind", "Tracks", "Commit"), mono_columns=(0, 2, 3),
                                  stretch_column=0)
        branches.add(self.branches)
        row = QHBoxLayout()
        self.checkout_button = QPushButton(icon("git-branch"), "Check out")
        self.checkout_button.clicked.connect(self.checkout_selected)
        self.new_button = QPushButton(icon("git-branch-plus"), "New branch…")
        self.new_button.clicked.connect(lambda: self.new_branch())
        row.addWidget(self.checkout_button)
        row.addWidget(self.new_button)
        row.addStretch(1)
        branches.add(row)
        box.addWidget(branches)

        remotes = Card("Remotes", "where pull and push go")
        self.remotes = DataTable(("Name", "URL"), mono_columns=(0, 1), stretch_column=1)
        remotes.add(self.remotes)
        row = QHBoxLayout()
        self.remote_button = QPushButton(icon("plus"), "Add remote…")
        self.remote_button.clicked.connect(lambda: self.add_remote())
        row.addWidget(self.remote_button)
        row.addStretch(1)
        remotes.add(row)
        box.addWidget(remotes)

        identity = Card("Identity", "the name and email on your commits")
        form = QFormLayout()
        self.name = QLineEdit()
        self.email = QLineEdit()
        self.everywhere = QCheckBox("Use for all repositories (git config --global)")
        form.addRow("Name", self.name)
        form.addRow("Email", self.email)
        form.addRow("", self.everywhere)
        identity.add(form)
        row = QHBoxLayout()
        self.identity_button = QPushButton(icon("save"), "Save identity")
        self.identity_button.clicked.connect(self.save_identity)
        row.addWidget(self.identity_button)
        row.addStretch(1)
        identity.add(row)
        box.addWidget(identity)
        box.addStretch(1)

    def refresh(self) -> None:
        git = self.control.git()
        self.control.run("Read branches", lambda: (git.branches(), git.remotes(), git.identity()),
                         self._show, self.message.emit)

    def _show(self, result) -> None:
        branches, remotes, (name, email) = result
        self.branch_list = branches
        self.branches.clear_rows()
        for branch in branches:
            kind = Pill("current", "ok") if branch.current else Pill("remote" if branch.remote else "local",
                                                                     "planned")
            self.branches.add_row([branch.name, kind, branch.upstream, branch.sha])
        self.remote_list = sorted(remotes.items())
        self.remotes.clear_rows()
        for remote, url in self.remote_list:
            self.remotes.add_row([remote, url])
        if not self.name.hasFocus():
            self.name.setText(name)
        if not self.email.hasFocus():
            self.email.setText(email)

    def _done(self, text: str):
        def done(_result) -> None:
            self.message.emit(text)
            self.control.changed.emit()
        return done

    def checkout_selected(self) -> None:
        rows = self.branches.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.branch_list):
            self.message.emit("Select a branch to check out.")
            return
        name = self.branch_list[rows[0].row()].name
        git = self.control.git()
        self.control.run("Check out", lambda: git.checkout(name), self._done(f"Checked out {name}"),
                         self.message.emit)

    def new_branch(self, name: str | None = None) -> None:
        if name is None:
            name, ok = QInputDialog.getText(self, "New branch", "Name of the new branch (it is checked out):")
            if not ok:
                return
        name = name.strip()
        if name:
            git = self.control.git()
            self.control.run("New branch", lambda: git.create_branch(name),
                             self._done(f"Made and checked out {name}"), self.message.emit)

    def add_remote(self, name: str | None = None, url: str | None = None) -> None:
        if name is None:
            name, ok = QInputDialog.getText(self, "Add remote", "Name:", text="origin")
            if not ok:
                return
        if url is None:
            url, ok = QInputDialog.getText(self, "Add remote", "URL (https://github.com/you/repo.git):")
            if not ok:
                return
        name, url = name.strip(), url.strip()
        if name and url:
            git = self.control.git()
            self.control.run("Add remote", lambda: git.add_remote(name, url),
                             self._done(f"Remote {name} → {url}"), self.message.emit)

    def save_identity(self) -> None:
        name, email = self.name.text().strip(), self.email.text().strip()
        if not name or "@" not in email:
            self.message.emit("Enter a name and an email address.")
            return
        git, everywhere = self.control.git(), self.everywhere.isChecked()
        self.control.run("Save identity", lambda: git.set_identity(name, email, everywhere=everywhere),
                         self._done(f"Commits will be signed {name} <{email}>"), self.message.emit)
