"""GitHub: connect an account with a token (checked, then kept in the OS keyring, never shown
again), your repositories (filter, clone, open in the browser), and this project on GitHub —
its open pull requests and issues, or publishing it to a new repository."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.common import CredentialError
from core.vcs import Git, Repo, github_slug

from ....services.source_control import SourceControl
from ...icons import icon
from ...widgets import Card, DataTable, Pill, label

__all__ = ["TOKEN_HELP", "GitHubView"]

TOKEN_HELP = "https://github.com/settings/personal-access-tokens/new"


class GitHubView(QWidget):
    message = Signal(str)
    cloned = Signal(Path)

    def __init__(self, control: SourceControl, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.repos: list[Repo] = []
        self.shown: list[Repo] = []
        self.items: list[str] = []
        self.slug: tuple[str, str] | None = None
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        account = Card("Account", "a fine-grained token with access to your repositories")
        self.account_state = label("", "CardCaption", wrap=True)
        account.add(self.account_state)
        row = QHBoxLayout()
        self.token_box = QLineEdit()
        self.token_box.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_box.setPlaceholderText("Paste a GitHub token (kept in the OS keyring)")
        self.connect_button = QPushButton(icon("log-in"), "Connect")
        self.connect_button.clicked.connect(lambda: self.connect_account())
        self.disconnect_button = QPushButton(icon("log-out"), "Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_account)
        self.help_button = QPushButton(icon("external-link"), "Make a token")
        self.help_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(TOKEN_HELP)))
        for widget in (self.token_box, self.connect_button, self.disconnect_button, self.help_button):
            row.addWidget(widget)
        row.setStretch(0, 1)
        account.add(row)
        box.addWidget(account)

        repos = Card("Your repositories", "most recently pushed first", padded=True)
        row = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name")
        self.filter.textChanged.connect(self._show_repos)
        self.clone_button = QPushButton(icon("cloud-download"), "Clone…")
        self.clone_button.clicked.connect(lambda: self.clone_selected())
        self.open_button = QPushButton(icon("external-link"), "Open on GitHub")
        self.open_button.clicked.connect(self.open_selected)
        row.addWidget(self.filter, 1)
        row.addWidget(self.clone_button)
        row.addWidget(self.open_button)
        repos.add(row)
        self.repo_table = DataTable(("Repository", "Visibility", "Pushed", "Language"),
                                    mono_columns=(0, 2), stretch_column=0)
        repos.add(self.repo_table)
        box.addWidget(repos)

        self.project = Card("This repository on GitHub", "")
        self.project_state = label("", "CardCaption", wrap=True)
        self.project.add(self.project_state)
        self.items_table = DataTable(("#", "Pull request or issue", "Author", "Updated"),
                                     mono_columns=(0, 3), stretch_column=1)
        self.items_table.cellDoubleClicked.connect(self._open_item)
        self.project.add(self.items_table)
        form = QFormLayout()
        self.repo_name = QLineEdit()
        self.private = QCheckBox("Private")
        self.private.setChecked(True)
        self.description = QLineEdit()
        form.addRow("New repository", self.repo_name)
        form.addRow("", self.private)
        form.addRow("Description", self.description)
        self.publish_form = QWidget()
        self.publish_form.setLayout(form)
        self.project.add(self.publish_form)
        row = QHBoxLayout()
        self.publish_button = QPushButton(icon("cloud-upload"), "Publish to GitHub")
        self.publish_button.setObjectName("Primary")
        self.publish_button.clicked.connect(self.publish)
        self.project_open = QPushButton(icon("external-link"), "Open on GitHub")
        self.project_open.clicked.connect(self._open_project)
        row.addWidget(self.publish_button)
        row.addWidget(self.project_open)
        row.addStretch(1)
        self.project.add(row)
        box.addWidget(self.project)
        box.addStretch(1)
        self.show_account(None)

    # The account ---------------------------------------------------------------------------------
    def refresh(self) -> None:
        if not self.control.token():
            self.show_account(None)
        else:
            hub = self.control.github()
            self.account_state.setText("Checking the token…")
            self.control.run("Read the GitHub account", lambda: (hub.user(), hub.repos(limit=300)),
                             self._connected, self._account_failed)
        self.refresh_project()

    def show_account(self, user: dict | None) -> None:
        connected = user is not None
        source = self.control.token_source()
        if connected:
            via = "the GITHUB_TOKEN environment variable" if source == "env" else "the OS keyring"
            self.account_state.setText(f"Connected as {user.get('login')} (token from {via}).")
        else:
            self.account_state.setText("Not connected. Public repositories clone without an account; your "
                                       "own list, private repositories and publishing need a token.")
        self.token_box.setVisible(not connected)
        self.connect_button.setVisible(not connected)
        self.help_button.setVisible(not connected)
        self.disconnect_button.setVisible(connected and source == "keyring")
        for widget in (self.filter, self.clone_button, self.open_button, self.repo_table):
            widget.setEnabled(connected)
        if not connected:
            self.repos = []
            self._show_repos()

    def connect_account(self, token: str | None = None) -> None:
        token = (token if token is not None else self.token_box.text()).strip()
        if not token:
            self.message.emit("Paste a token first.")
            return
        hub = self.control.github(token)

        def store(result) -> None:
            try:
                self.control.store_token(token)
            except CredentialError as exc:
                self.message.emit(str(exc))
                return
            self.token_box.clear()
            self._connected(result)
            self.message.emit(f"Connected to GitHub as {result[0]['login']}")

        self.account_state.setText("Checking the token…")
        self.control.run("Connect GitHub", lambda: (hub.user(), hub.repos(limit=300)), store,
                         self._account_failed)

    def _account_failed(self, error: str) -> None:
        self.account_state.setText(error)
        self.message.emit(error)

    def disconnect_account(self) -> None:
        try:
            self.control.forget_token()
        except CredentialError as exc:
            self.message.emit(str(exc))
            return
        self.show_account(None)
        self.message.emit("Disconnected from GitHub; the token was removed from the keyring.")

    def _connected(self, result) -> None:
        user, repos = result
        self.show_account(user)
        self.repos = repos
        self._show_repos()
        self.refresh_project()

    # Repositories ------------------------------------------------------------------------------------
    def _show_repos(self) -> None:
        needle = self.filter.text().strip().lower()
        self.shown = [r for r in self.repos if needle in r.full_name.lower()]
        self.repo_table.clear_rows()
        for repo in self.shown[:200]:
            self.repo_table.add_row([repo.full_name, Pill("private", "warn") if repo.private else
                                     Pill("public", "ok"), repo.updated_at, repo.language])

    def _selected_repo(self) -> Repo | None:
        rows = self.repo_table.selectionModel().selectedRows()
        return self.shown[rows[0].row()] if rows and rows[0].row() < len(self.shown) else None

    def open_selected(self) -> None:
        repo = self._selected_repo()
        if repo:
            QDesktopServices.openUrl(QUrl(repo.html_url))

    def clone_selected(self, parent_folder: Path | None = None) -> None:
        repo = self._selected_repo()
        if repo is None:
            self.message.emit("Select a repository to clone.")
            return
        if parent_folder is None:
            chosen = QFileDialog.getExistingDirectory(self, "Clone into which folder?", str(Path.home()))
            if not chosen:
                return
            parent_folder = Path(chosen)
        target = parent_folder / repo.full_name.split("/")[-1]
        token = self.control.token()
        self.message.emit(f"Cloning {repo.full_name} into {target}…")
        self.control.run("Clone", lambda: Git.clone(repo.clone_url, target, token=token).root,
                         lambda root: (self.message.emit(f"Cloned into {root}"), self.cloned.emit(root)),
                         self.message.emit)

    # This project --------------------------------------------------------------------------------------
    def refresh_project(self) -> None:
        if self.control.root is None:
            self.project.setVisible(False)
            return
        self.project.setVisible(True)
        git = self.control.git()
        self.control.run("Read remotes", git.remotes, self._show_project, self.message.emit)

    def _show_project(self, remotes: dict[str, str]) -> None:
        origin = remotes.get("origin", "")
        self.slug = github_slug(origin) if origin else None
        self.items_table.clear_rows()
        self.items = []
        published = self.slug is not None
        self.publish_form.setVisible(not published and not origin)
        self.publish_button.setVisible(not published and not origin)
        self.project_open.setVisible(published)
        self.items_table.setVisible(published)
        connected = bool(self.control.token())
        if published:
            self.project_state.setText(f"origin is github.com/{'/'.join(self.slug)}.")
            if connected:
                owner, name = self.slug
                hub = self.control.github()
                self.control.run("Read pull requests and issues",
                                 lambda: (hub.pulls(owner, name), hub.issues(owner, name)), self._show_items,
                                 lambda error: self.project_state.setText(error))
        elif origin:
            self.project_state.setText(f"origin is {origin} (not GitHub).")
        else:
            root = self.control.root
            self.repo_name.setText(self.repo_name.text() or (root.name if root else ""))
            self.project_state.setText("This repository has no origin yet: publish it to a new GitHub "
                                       "repository (needs a connected account)." if connected else
                                       "Connect an account above to publish this repository to GitHub.")
            self.publish_button.setEnabled(connected)

    def _show_items(self, result) -> None:
        pulls, issues = result
        self.items_table.clear_rows()
        self.items = []
        for kind, entries in (("PR", pulls), ("Issue", issues)):
            for item in entries:
                self.items_table.add_row([f"#{item.number}", f"{kind}: {item.title}", item.author,
                                          item.updated_at])
                self.items.append(item.html_url)
        if not self.items:
            self.project_state.setText(self.project_state.text() + " No open pull requests or issues.")

    def _open_item(self, row: int, _column: int) -> None:
        if 0 <= row < len(self.items):
            QDesktopServices.openUrl(QUrl(self.items[row]))

    def _open_project(self) -> None:
        if self.slug:
            QDesktopServices.openUrl(QUrl(f"https://github.com/{'/'.join(self.slug)}"))

    def publish(self) -> None:
        name = self.repo_name.text().strip()
        if not name:
            self.message.emit("Name the new repository first.")
            return
        hub, git = self.control.github(), self.control.git()
        private, description = self.private.isChecked(), self.description.text().strip()

        def work() -> str:
            repo = hub.create_repo(name, private=private, description=description)
            git.add_remote("origin", repo.clone_url)
            if git.status().head:
                git.push()
            return repo.full_name

        self.publish_button.setEnabled(False)
        self.message.emit(f"Publishing to github.com/…/{name}…")
        self.control.run("Publish to GitHub", work,
                         lambda full: (self.message.emit(f"Published to github.com/{full}"),
                                       self.control.changed.emit()),
                         lambda error: (self.message.emit(error), self.publish_button.setEnabled(True)))
