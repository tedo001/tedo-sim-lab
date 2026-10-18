"""Git & GitHub: the Code page's project (or any folder) under version control — changes and
commits, history, branches and remotes, fetch / pull / push — and a GitHub account: your
repositories, cloning, and publishing a project. A folder that is not a repository yet can be
made one, or a repository cloned into it. Every git and GitHub call runs in the background."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from core.vcs import Git, RepoStatus

from ....services.context import AppContext
from ....services.source_control import SourceControl
from ...icons import icon
from ...widgets import Card, Page, PathLabel, Segmented, label
from .branches import BranchesView
from .changes import ChangesView
from .github_view import GitHubView
from .history import HistoryView

__all__ = ["SourceControlPage"]


class SourceControlPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Git & GitHub", "commit, branch, sync, and your GitHub repositories", parent)
        self.ctx = ctx
        self.control = SourceControl(ctx.paths, ctx.credentials, ctx.jobs, self)
        self.folder = ctx.terminals.default_folder()
        self.chosen = False  # the person picked a folder: stop following the Code page
        self.status: RepoStatus | None = None
        open_button = QPushButton(icon("folder-open"), "Open folder…")
        open_button.clicked.connect(self.choose_folder)
        clone_button = QPushButton(icon("cloud-download"), "Clone…")
        clone_button.clicked.connect(lambda: self.clone_url())
        refresh = QPushButton(icon("refresh-cw"), "Refresh")
        refresh.clicked.connect(self.refresh)
        for widget in (open_button, clone_button, refresh):
            self.head.add_action(widget)

        bar = Card("Repository", "")
        row = QHBoxLayout()
        self.path_label = PathLabel(self.folder)
        row.addWidget(self.path_label, 1)
        self.branch_box = QComboBox()
        self.branch_box.setMinimumWidth(160)
        self.branch_box.setToolTip("The current branch: choose another to check it out")
        self.branch_box.activated.connect(self._switch_branch)
        row.addWidget(self.branch_box)
        self.sync_label = label("", "Mono")
        row.addWidget(self.sync_label)
        self.fetch_button = QPushButton(icon("refresh-cw"), "Fetch")
        self.pull_button = QPushButton(icon("arrow-down-to-line"), "Pull")
        self.push_button = QPushButton(icon("arrow-up-from-line"), "Push")
        self.terminal_button = QPushButton(icon("square-terminal"), "Terminal here")
        self.terminal_button.setToolTip("A terminal in this repository (for git commands, SSH passphrases)")
        for button, slot in ((self.fetch_button, self.fetch), (self.pull_button, self.pull),
                             (self.push_button, self.push), (self.terminal_button, self.open_terminal)):
            button.clicked.connect(slot)
            row.addWidget(button)
        bar.add(row)
        self.init_row = QWidget()
        init_box = QHBoxLayout(self.init_row)
        init_box.setContentsMargins(0, 0, 0, 0)
        self.init_hint = label("", "CardCaption", wrap=True)
        self.init_button = QPushButton(icon("git-branch-plus"), "Make it a repository")
        self.init_button.setObjectName("Primary")
        self.init_button.clicked.connect(self.init_repository)
        init_box.addWidget(self.init_hint, 1)
        init_box.addWidget(self.init_button)
        bar.add(self.init_row)
        self.message = label("", "CardCaption", wrap=True)
        self.message.hide()
        bar.add(self.message)
        self.body.addWidget(bar)

        self.views = Segmented([("changes", "Changes"), ("history", "History"),
                                ("branches", "Branches and remotes"), ("github", "GitHub")])
        self.views.changed.connect(self._view_changed)
        self.body.addWidget(self.views)
        self.stack = QStackedWidget()
        self.changes = ChangesView(self.control)
        self.history = HistoryView(self.control)
        self.branches = BranchesView(self.control)
        self.github = GitHubView(self.control)
        self.pages = {"changes": self.changes, "history": self.history, "branches": self.branches,
                      "github": self.github}
        for view in self.pages.values():
            view.message.connect(self.say)
            self.stack.addWidget(view)
        self.github.cloned.connect(self.open_folder)
        self.stack.setMinimumHeight(520)
        self.body.addWidget(self.stack, 1)
        self.control.changed.connect(self.refresh)
        ctx.terminals.project_changed.connect(self._project_changed)
        self.open_folder(self.folder, chosen=False)

    # The folder -------------------------------------------------------------------------------------
    def _project_changed(self, root: Path | None) -> None:
        if not self.chosen and root is not None:
            self.open_folder(root, chosen=False)

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Open a folder", str(self.folder))
        if chosen:
            self.open_folder(Path(chosen))

    def open_folder(self, folder: Path, *, chosen: bool = True) -> None:
        self.folder, self.chosen = Path(folder), self.chosen or chosen
        self.path_label.set_full_text(str(self.folder))
        self.control.root = None
        if not self.control.git_available:
            self._show_repository(False, "git is not installed: get it from https://git-scm.com, then "
                                         "press Refresh.")
            return
        folder = self.folder
        self.control.run("Find the repository", lambda: Git.find_root(folder), self._found, self.say)

    def _found(self, root: Path | None) -> None:
        if root is None:
            self._show_repository(False, f"{self.folder} is not a git repository yet.")
            return
        self.control.root = root
        self.path_label.set_full_text(str(root))
        self._show_repository(True, "")
        self.refresh()

    def _show_repository(self, present: bool, hint: str) -> None:
        self.init_row.setVisible(not present)
        self.init_hint.setText(hint)
        self.init_button.setVisible(self.control.git_available)
        for widget in (self.branch_box, self.fetch_button, self.pull_button, self.push_button):
            widget.setEnabled(present)
        self.terminal_button.setEnabled(True)
        if not present:
            self.status = None
            self.changes.show_status(None)
            self.history.show_commits([])
            self.sync_label.setText("")
            self.branch_box.clear()
        self.github.refresh_project()

    # Reading ----------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        if self.control.root is None:
            if self.folder is not None and self.control.git_available:
                self.open_folder(self.folder, chosen=False)
            return
        git = self.control.git()
        self.control.run("Read git status", lambda: (git.status(), git.branches()), self._show, self.say)
        current = self.views.current
        if current != "changes":
            self._view_changed(current)

    def _show(self, result) -> None:
        status, branches = result
        self.status = status
        self.changes.show_status(status)
        self.branch_box.clear()
        local = [b.name for b in branches if not b.remote]
        names = local or [status.branch or "(no branch)"]
        self.branch_box.addItems(names)
        if status.branch in names:
            self.branch_box.setCurrentText(status.branch)
        parts = [status.head or "no commits"]
        if status.upstream:
            parts.append(f"↑{status.ahead} ↓{status.behind} {status.upstream}")
        elif status.branch:
            parts.append("not published")
        self.sync_label.setText(" · ".join(parts))
        self.push_button.setText(f"Push ({status.ahead})" if status.ahead else "Push")
        self.pull_button.setText(f"Pull ({status.behind})" if status.behind else "Pull")

    def _view_changed(self, key: str) -> None:
        self.stack.setCurrentWidget(self.pages[key])
        if key in ("history", "branches", "github") and (self.control.root or key == "github"):
            self.pages[key].refresh()

    def say(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    # Acting ------------------------------------------------------------------------------------------------
    def _network(self, title: str, work, done_text: str) -> None:
        self.say(f"{title}…")
        self.control.run(title, work, lambda _out: (self.say(done_text), self.refresh()), self.say)

    def fetch(self) -> None:
        git = self.control.git()
        self._network("Fetch", git.fetch, "Fetched from every remote.")

    def pull(self) -> None:
        git = self.control.git()
        self._network("Pull", git.pull, "Pulled (fast-forward).")

    def push(self) -> None:
        git = self.control.git()
        self._network("Push", git.push, "Pushed.")

    def _switch_branch(self, index: int) -> None:
        name = self.branch_box.itemText(index)
        if self.status is not None and name and name != self.status.branch:
            git = self.control.git()
            self.control.run("Check out", lambda: git.checkout(name),
                             lambda _r: (self.say(f"Checked out {name}"), self.refresh()), self.say)

    def init_repository(self) -> None:
        folder = self.folder
        self.control.run("Make a repository", lambda: Git.init(folder).root,
                         lambda root: (self.say(f"Made a git repository in {root}"), self._found(root)),
                         self.say)

    def clone_url(self, url: str | None = None, parent_folder: Path | None = None) -> None:
        if url is None:
            url, ok = QInputDialog.getText(self, "Clone a repository", "Repository URL:")
            if not ok or not url.strip():
                return
        if parent_folder is None:
            chosen = QFileDialog.getExistingDirectory(self, "Clone into which folder?", str(Path.home()))
            if not chosen:
                return
            parent_folder = Path(chosen)
        name = url.rstrip("/").split("/")[-1].removesuffix(".git") or "repository"
        target, token = parent_folder / name, self.control.token()
        self.say(f"Cloning into {target}…")
        self.control.run("Clone", lambda: Git.clone(url.strip(), target, token=token).root,
                         lambda root: (self.say(f"Cloned into {root}"), self.open_folder(root)), self.say)

    def open_terminal(self) -> None:
        self.ctx.open_terminal(self.control.root or self.folder)
