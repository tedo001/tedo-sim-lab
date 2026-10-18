"""The main window: top bar, sidebar, and a stack of pages built on first visit.

Layout controls: Ctrl+B collapses the sidebar to an icon rail; Ctrl+\\ opens a split
view with a second page beside the first; Ctrl+` opens the terminal panel under the pages. All
are remembered between sessions.
"""

from __future__ import annotations

import platform
from pathlib import Path

from PySide6.QtCore import Qt, Signal, qVersion
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .navigation import NAV, page
from .services.context import AppContext
from .services.ui_state import UiState
from .ui.pages import build_page
from .ui.shell import Sidebar, SplitPane, TopBar
from .ui.shell.search_index import search_entries
from .ui.shell.terminal_panel import TerminalPanel

__all__ = ["MainWindow", "WINDOW_TITLE"]

WINDOW_TITLE = "TEDO AI Research Lab"


class MainWindow(QMainWindow):
    page_changed = Signal(str)

    def __init__(self, ctx: AppContext, ui_state: UiState | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.ui_state = ui_state or UiState()
        ctx.navigate = self.navigate
        ctx.open_file = self.open_file
        ctx.open_terminal = self.open_terminal
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)
        self._pages: dict[str, QWidget] = {}
        self._current = ""

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.top_bar = TopBar()
        self.top_bar.page_requested.connect(self.navigate)
        self.top_bar.entry_requested.connect(self.open_entry)
        self.top_bar.sidebar_toggle_requested.connect(self.toggle_sidebar)
        self.top_bar.split_toggle_requested.connect(self.toggle_split)
        self.top_bar.terminal_toggle_requested.connect(self.toggle_terminal)
        column.addWidget(self.top_bar)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.page_requested.connect(self.navigate)
        self.stack = QStackedWidget()
        self.split_pane = SplitPane(lambda page_id: build_page(page_id, self.ctx))
        self.split_pane.close_requested.connect(lambda: self.set_split_open(False))
        self.split_pane.page_changed.connect(self._remember_split_page)
        self.split_pane.hide()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("MainSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.stack)
        self.splitter.addWidget(self.split_pane)
        self.terminal_panel = TerminalPanel(ctx)
        self.terminal_panel.close_requested.connect(lambda: self.set_terminal_open(False))
        self.terminal_panel.hide()
        self.vertical = QSplitter(Qt.Orientation.Vertical)
        self.vertical.setObjectName("TerminalSplitter")
        self.vertical.setChildrenCollapsible(False)
        self.vertical.setHandleWidth(1)
        self.vertical.addWidget(self.splitter)
        self.vertical.addWidget(self.terminal_panel)
        self.vertical.setStretchFactor(0, 1)
        row.addWidget(self.sidebar)
        row.addWidget(self.vertical, 1)
        column.addLayout(row, 1)
        self.setCentralWidget(central)

        self.status_label = QLabel(f"workspace {ctx.paths.workspace}")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(QLabel(
            f"Python {platform.python_version()} · Qt {qVersion()} · v{__version__}"))
        self.statusBar().setSizeGripEnabled(False)

        for keys, slot in (("Ctrl+B", self.toggle_sidebar), ("Ctrl+\\", self.toggle_split),
                           ("Ctrl+`", self.toggle_terminal)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)

        ctx.hardware.sampled.connect(self.top_bar.show_sample)
        self.refresh_search()
        ctx.experiments.run_changed.connect(lambda _: self.refresh_search())
        ctx.downloads.imported.connect(lambda *_: self.refresh_search())
        ctx.hardware.start()
        self.navigate(NAV[0].id)
        self._restore_layout()

    def page_widget(self, page_id: str) -> QWidget:
        """The page for ``page_id``, built the first time it is asked for."""
        if page_id not in self._pages:
            widget = build_page(page_id, self.ctx)
            widget.setProperty("pageId", page_id)
            self._pages[page_id] = widget
            self.stack.addWidget(widget)
        return self._pages[page_id]

    def navigate(self, page_id: str) -> None:
        spec = page(page_id)  # KeyError for an unknown id, on purpose
        self.stack.setCurrentWidget(self.page_widget(page_id))
        self.sidebar.set_current(page_id)
        self.setWindowTitle(f"{spec.title} — {WINDOW_TITLE}")
        if page_id != self._current:
            self._current = page_id
            self.page_changed.emit(page_id)

    def refresh_search(self) -> None:
        self.top_bar.set_entries(search_entries(self.ctx))

    def open_entry(self, key: str) -> None:
        """Open a search entry: a page, a dataset card, a model card or a run."""
        kind, _, ident = key.partition(":")
        if kind == "page":
            self.navigate(ident)
        elif kind == "dataset":
            self.navigate("dataset_hub")
            self.page_widget("dataset_hub").show_view("catalogue").select(ident)
        elif kind == "model":
            self.navigate("model_zoo")
            self.page_widget("model_zoo").select(ident)
        elif kind == "run":
            self.ctx.experiments.show(ident, self.navigate)

    def current_page_id(self) -> str:
        return self._current

    def open_file(self, path: Path, line: int = 1) -> None:
        """Open ``path`` at ``line`` in the Code page (from a terminal's Ctrl+click, a log, ...)."""
        self.navigate("code")
        self.page_widget("code").open_at(Path(path), int(line))

    # Layout -----------------------------------------------------------------
    def sidebar_collapsed(self) -> bool:
        return self.sidebar.collapsed()

    def set_sidebar_collapsed(self, collapsed: bool) -> None:
        self.sidebar.set_collapsed(collapsed)
        self.top_bar.show_sidebar_collapsed(collapsed)
        self.ui_state.sidebar_collapsed = collapsed

    def toggle_sidebar(self) -> None:
        self.set_sidebar_collapsed(not self.sidebar_collapsed())

    def split_open(self) -> bool:
        return not self.split_pane.isHidden()

    def set_split_open(self, open_: bool, page_id: str | None = None) -> None:
        """Show or hide the second pane; ``page_id`` chooses what it shows."""
        if open_:
            wanted = page_id or self.split_pane.current_page_id() or self.ui_state.split_page
            try:
                self.split_pane.set_page(wanted)
            except KeyError:  # a page id remembered from an older build
                self.split_pane.set_page("hardware")
            if self.split_pane.isHidden():
                self.split_pane.show()
                half = max(self.splitter.width() // 2, 200)
                self.splitter.setSizes([half, half])
        else:
            self.split_pane.hide()
        self.top_bar.show_split_open(open_)
        self.ui_state.split_open = open_

    def toggle_split(self) -> None:
        self.set_split_open(not self.split_open())

    def open_terminal(self, folder: Path) -> None:
        """A new terminal in ``folder`` on the Terminal page."""
        self.navigate("terminal")
        self.page_widget("terminal").open_in(Path(folder))

    def terminal_open(self) -> bool:
        return not self.terminal_panel.isHidden()

    def set_terminal_open(self, open_: bool) -> None:
        """Show or hide the terminal panel; opening it starts a shell when it has none."""
        if open_:
            if self.terminal_panel.isHidden():
                self.terminal_panel.show()
                total = max(self.vertical.height(), 400)
                state = self.ui_state.terminal_splitter_state
                if state is None or not self.vertical.restoreState(state):
                    self.vertical.setSizes([total - 280, 280])
            self.terminal_panel.open_if_empty()
        else:
            if not self.terminal_panel.isHidden():
                self.ui_state.terminal_splitter_state = self.vertical.saveState()
            self.terminal_panel.hide()
        self.top_bar.show_terminal_open(open_)
        self.ui_state.terminal_open = open_

    def toggle_terminal(self) -> None:
        self.set_terminal_open(not self.terminal_open())

    def _remember_split_page(self, page_id: str) -> None:
        self.ui_state.split_page = page_id

    def _restore_layout(self) -> None:
        self.set_sidebar_collapsed(self.ui_state.sidebar_collapsed)
        if self.ui_state.terminal_open:
            self.set_terminal_open(True)
        if self.ui_state.split_open:
            self.set_split_open(True)
            state = self.ui_state.splitter_state
            if state is not None:
                self.splitter.restoreState(state)

    def _save_layout(self) -> None:
        if self.split_open():
            self.ui_state.splitter_state = self.splitter.saveState()
        if self.terminal_open():
            self.ui_state.terminal_splitter_state = self.vertical.saveState()
        self.ui_state.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ask before abandoning running jobs; then stop them cleanly."""
        active = self.ctx.jobs.active()
        if active and self.isVisible() and not self._confirm_quit(len(active)):
            event.ignore()
            return
        unsaved = [name for widget in self._all_pages() for name in getattr(widget, "unsaved_work", list)()]
        if unsaved and self.isVisible() and not self._confirm_unsaved(unsaved):
            event.ignore()
            return
        self._save_layout()
        self.ctx.jobs.shutdown()
        super().closeEvent(event)

    def _all_pages(self) -> list[QWidget]:
        return [*self._pages.values(), *self.split_pane.pages(), self.terminal_panel]

    def _confirm_unsaved(self, names: list[str]) -> bool:
        from PySide6.QtWidgets import QMessageBox
        listed = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        answer = QMessageBox.question(self, "Unsaved changes",
                                      f"{listed} {'has' if len(names) == 1 else 'have'} unsaved changes. "
                                      "Quit without saving?")
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_quit(self, count: int) -> bool:
        from PySide6.QtWidgets import QMessageBox
        answer = QMessageBox.question(
            self, "Jobs are still running",
            f"{count} job(s) are queued or running. Quitting cancels them (checkpoints already "
            "written are kept). Quit anyway?")
        return answer == QMessageBox.StandardButton.Yes
