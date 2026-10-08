"""The main window: top bar, sidebar, and a stack of pages built on first visit."""

from __future__ import annotations

import platform

from PyQt6.QtCore import QT_VERSION_STR, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from . import __version__
from .navigation import NAV, page
from .services.context import AppContext
from .ui.pages import build_page
from .ui.shell import Sidebar, TopBar

__all__ = ["MainWindow", "WINDOW_TITLE"]

WINDOW_TITLE = "TEDO AI Research Lab"


class MainWindow(QMainWindow):
    page_changed = pyqtSignal(str)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        ctx.navigate = self.navigate
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
        column.addWidget(self.top_bar)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.page_requested.connect(self.navigate)
        self.stack = QStackedWidget()
        row.addWidget(self.sidebar)
        row.addWidget(self.stack, 1)
        column.addLayout(row, 1)
        self.setCentralWidget(central)

        self.status_label = QLabel(f"workspace {ctx.paths.workspace}")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(QLabel(
            f"Python {platform.python_version()} · Qt {QT_VERSION_STR} · v{__version__}"))
        self.statusBar().setSizeGripEnabled(False)

        self.navigate(NAV[0].id)

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

    def current_page_id(self) -> str:
        return self._current
