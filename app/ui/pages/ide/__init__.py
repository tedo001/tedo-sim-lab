"""Code: a small IDE for a whole project. Open a folder; browse and manage its files; edit with
syntax colouring; run the current file (F5) with the experiment Python, the project on
``PYTHONPATH`` and the project folder as working directory; read the output (double-click a
traceback line to jump there); run it in a terminal instead (Ctrl+F5, it can read input); debug
it with pdb (Shift+F9; F9 or a click in the gutter sets breakpoints) while the editor follows;
use a terminal in the project; see its git state."""

from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.common import experiment_python

from ....services.context import AppContext
from ....services.project import RUNNABLE, ProjectRunner, RecentProjects
from ...icons import icon
from ...widgets import Page, PathLabel, label
from .console import BottomPanel
from .debugging import Debugger
from .editors import EditorTabs, FindBar
from .tree import ProjectTree

__all__ = ["IdePage"]


class IdePage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *,
                 recent: RecentProjects | None = None) -> None:
        super().__init__("Code", "open a project folder, edit and run its files", parent)
        self.ctx = ctx
        self.recent = recent or RecentProjects()
        self.root: Path | None = None
        self.runner = ProjectRunner(ctx.config, ctx.paths.workspace, self)

        open_button = QPushButton(icon("folder-open"), "Open folder…")
        open_button.clicked.connect(self._choose_folder)
        self.recent_button = QPushButton("Recent")
        self.recent_button.setMenu(QMenu(self.recent_button))
        self.recent_button.menu().aboutToShow.connect(self._fill_recent)
        for widget in (open_button, self.recent_button):
            self.head.add_action(widget)

        top = QHBoxLayout()
        self.project_label = PathLabel("")
        top.addWidget(label("Project", "KvKey"))
        top.addWidget(self.project_label, 1)
        self.args = QLineEdit()
        self.args.setPlaceholderText("Arguments for Run")
        self.args.setMaximumWidth(260)
        top.addWidget(self.args)
        self.save_button = QPushButton(icon("save"), "Save")
        self.save_button.clicked.connect(lambda: self.tabs.save())
        self.run_button = QPushButton(icon("play"), "Run")
        self.run_button.setObjectName("Primary")
        self.run_button.setToolTip("Run the current file (F5)")
        self.run_button.clicked.connect(self.run_current)
        self.stop_button = QPushButton(icon("square"), "Stop")
        self.stop_button.setToolTip("Stop (Shift+F5)")
        self.stop_button.clicked.connect(self.stop)
        self.terminal_run_button = QPushButton(icon("square-terminal"), "")
        self.terminal_run_button.setToolTip("Run the current file in a terminal, where it can read "
                                            "input (Ctrl+F5)")
        self.terminal_run_button.clicked.connect(lambda: self.debug_current(debug=False))
        self.debug_button = QPushButton(icon("bug"), "Debug")
        self.debug_button.setToolTip("Debug the current file with pdb (Shift+F9); F9 or a click in "
                                     "the gutter sets a breakpoint")
        self.debug_button.clicked.connect(self.debug_current)
        for widget in (self.save_button, self.run_button, self.terminal_run_button, self.debug_button,
                       self.stop_button):
            top.addWidget(widget)
        self.body.addLayout(top)

        self.tree = ProjectTree()
        self.tabs = EditorTabs()
        self.find_bar = FindBar(self.tabs)
        self.find_bar.hide()
        self.status = label("Open a file from the tree on the left, or make one (right-click).",
                            "CardCaption", wrap=True)
        editor_column = QWidget()
        column = QVBoxLayout(editor_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.tabs, 1)
        column.addWidget(self.find_bar)
        column.addWidget(self.status)
        across = QSplitter(Qt.Orientation.Horizontal)
        across.addWidget(self.tree)
        across.addWidget(editor_column)
        across.setStretchFactor(1, 1)
        across.setSizes([240, 760])
        self.panel = BottomPanel(ctx)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(across)
        self.splitter.addWidget(self.panel)
        self.splitter.setSizes([520, 220])
        self.splitter.setMinimumHeight(640)
        self.splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.body.addWidget(self.splitter, 1)

        self.tree.open_requested.connect(self.open_file)
        self.tree.moved.connect(self.tabs.follow)
        self.tree.message.connect(self.status.setText)
        self.tabs.message.connect(self.status.setText)
        self.tabs.current_file.connect(lambda _p: self._update())
        self.panel.jump.connect(self.open_at)
        self.runner.output.connect(self.panel.output.write)
        self.runner.started.connect(self._started)
        self.runner.finished.connect(self._finished)
        self.debugger = Debugger(ctx, self.tabs, self.panel, self)
        self.debugger.message.connect(self.status.setText)
        self.debugger.active_changed.connect(lambda _on: self._update())

        for keys, slot in (("Ctrl+S", lambda: self.tabs.save()), ("Ctrl+Shift+S", self.tabs.save_all),
                           ("F5", self.run_current), ("Shift+F5", self.stop),
                           ("Ctrl+F5", lambda: self.debug_current(debug=False)),
                           ("Shift+F9", self.debug_current),
                           ("Ctrl+F", self._find), ("Escape", self.find_bar.hide),
                           ("Ctrl+W", lambda: self.tabs.close_tab(self.tabs.currentIndex()))):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        recent = self.recent.all()
        self.open_project(recent[0] if recent else ctx.paths.workspace, remember=False)

    # Project ------------------------------------------------------------------------------------
    def open_project(self, folder: Path, *, remember: bool = True) -> None:
        folder = folder.resolve()
        if not folder.is_dir():
            self.status.setText(f"{folder} is not a folder")
            return
        self.root = folder
        self.tree.set_root(folder)
        self.project_label.set_full_text(str(folder))
        self.panel.set_root(folder)
        self.ctx.terminals.set_project(folder)
        if remember:
            self.recent.add(folder)
        self._update()

    def _choose_folder(self) -> None:
        start = str(self.root or Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open a project folder", start)
        if folder:
            self.open_project(Path(folder))

    def _fill_recent(self) -> None:
        menu = self.recent_button.menu()
        menu.clear()
        for folder in self.recent.all():
            menu.addAction(str(folder), lambda f=folder: self.open_project(f))
        if menu.isEmpty():
            menu.addAction("No recent projects").setEnabled(False)

    # Files --------------------------------------------------------------------------------------
    def open_file(self, path: Path) -> None:
        if self.tabs.open(path) is not None:
            self.status.setText(str(path))

    def open_at(self, path: Path, line: int) -> None:
        editor = self.tabs.open(path)
        if editor is not None:
            editor.go_to_line(line)

    def _find(self) -> None:
        editor = self.tabs.current_editor()
        self.find_bar.show_for(editor.textCursor().selectedText() if editor else "")

    def unsaved_work(self) -> list[str]:
        """Files with unsaved changes (the window asks before quitting)."""
        return [path.name for path in self.tabs.unsaved()] + self.panel.terminals.unsaved_work()

    # Running ------------------------------------------------------------------------------------
    def run_current(self) -> bool:
        path = self.tabs.current_path()
        if path is None or self.root is None:
            self.status.setText("Open the file to run first.")
            return False
        if self.tabs.current_editor().document().isModified() and not self.tabs.save():
            return False
        try:
            args = shlex.split(self.args.text(), posix=True)
        except ValueError as exc:
            self.status.setText(f"Arguments: {exc}")
            return False
        self.panel.setCurrentIndex(0)
        try:
            self.runner.run(path, self.root, args)
        except ValueError as exc:
            self.panel.output.write(f"{exc}\n", "stderr")
            self.status.setText(str(exc))
            return False
        return True

    def debug_current(self, debug: bool = True) -> bool:
        """Debug the current file with pdb, or run it in a terminal (``debug=False``)."""
        path = self.tabs.current_path()
        if path is None or self.root is None:
            self.status.setText(f"Open the file to {'debug' if debug else 'run'} first.")
            return False
        if self.tabs.current_editor().document().isModified() and not self.tabs.save():
            return False
        try:
            args = shlex.split(self.args.text(), posix=True)
        except ValueError as exc:
            self.status.setText(f"Arguments: {exc}")
            return False
        return self.debugger.start(path, self.root, args, debug=debug)

    def stop(self) -> None:
        if self.runner.running:
            self.runner.stop()
        elif self.debugger.active:
            self.debugger.stop()

    def _started(self, command: str) -> None:
        self.panel.output.write(f"\n$ {command}\n", "info")
        self._update()

    def _finished(self, code: int, stopped: bool) -> None:
        text = "Stopped" if stopped else f"Finished with exit code {code}"
        self.panel.output.write(f"[{text}]\n", "info")
        self.status.setText(text)
        self.panel.refresh_git()
        self._update()

    def _update(self) -> None:
        path = self.tabs.current_path()
        runnable = path is not None and path.suffix.lower() in RUNNABLE
        running = self.runner.running
        self.run_button.setEnabled(runnable and not running)
        self.terminal_run_button.setEnabled(runnable)
        self.debug_button.setEnabled(path is not None and path.suffix.lower() in (".py", ".pyw"))
        python = experiment_python(self.ctx.config)
        self.run_button.setToolTip("Run the current file (F5)" if runnable else
                                   "Open a Python, shell, PowerShell or batch file to run it")
        if runnable and path.suffix.lower().startswith(".py") and not python:
            self.run_button.setToolTip("Set up the experiment Python first (Settings)")
        self.stop_button.setEnabled(running or self.debugger.active)
        self.save_button.setEnabled(path is not None)
