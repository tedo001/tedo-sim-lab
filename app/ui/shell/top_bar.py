"""The title row: sidebar toggle, wordmark, Ctrl+K page search, split view and settings.

Search covers pages in this build; datasets, models and runs join it once
their registries exist (build phases 2 and 7).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, QStringListModel, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QCompleter, QFrame, QHBoxLayout, QLineEdit, QToolButton, QWidget

from ... import __version__
from ...navigation import NAV
from ..icons import icon
from ..theme.tokens import COLORS, SIZES
from ..widgets import label

__all__ = ["TopBar"]


def _tool_button(name: str, tooltip: str) -> QToolButton:
    button = QToolButton()
    button.setObjectName("TopBarButton")
    button.setIcon(icon(name, COLORS["text_dim"], 18))
    button.setIconSize(QSize(18, 18))
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip.split(" (")[0])
    return button


class TopBar(QFrame):
    page_requested = Signal(str)
    sidebar_toggle_requested = Signal()
    split_toggle_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(SIZES["topbar_height"])
        self._titles = {spec.title: spec.id for spec in NAV}

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 0, 12, 0)
        row.setSpacing(10)
        self.sidebar_button = _tool_button("panel-left-close", "Collapse sidebar (Ctrl+B)")
        self.sidebar_button.clicked.connect(self.sidebar_toggle_requested.emit)
        row.addWidget(self.sidebar_button)
        row.addWidget(label("TEDO", "Wordmark"))
        row.addWidget(label("AI RESEARCH LAB", "WordmarkSub"))
        row.addWidget(label(f"v{__version__}", "VersionTag"), 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        self.search = QLineEdit()
        self.search.setObjectName("GlobalSearch")
        self.search.setPlaceholderText("Go to page…   Ctrl+K")
        self.search.setFixedWidth(360)
        self.search.setClearButtonEnabled(True)
        self.search.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # Ctrl+K or a click, not on startup
        self.search.addAction(QAction(icon("search", COLORS["text_faint"], 14), "", self.search),
                              QLineEdit.ActionPosition.LeadingPosition)
        completer = QCompleter(QStringListModel(sorted(self._titles)), self.search)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.popup().setObjectName("Completer")
        completer.activated.connect(self._go)
        self.search.setCompleter(completer)
        self.search.returnPressed.connect(lambda: self._go(self.search.text()))
        row.addWidget(self.search)
        row.addStretch(1)

        self.split_button = _tool_button("columns-2", "Split view (Ctrl+\\)")
        self.split_button.setCheckable(True)
        self.split_button.clicked.connect(lambda _=False: self.split_toggle_requested.emit())
        row.addWidget(self.split_button)
        self.settings_button = _tool_button("settings", "Settings")
        self.settings_button.clicked.connect(lambda: self.page_requested.emit("settings"))
        row.addWidget(self.settings_button)

        search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        search_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        search_shortcut.activated.connect(self.focus_search)

    def show_sidebar_collapsed(self, collapsed: bool) -> None:
        name, tip = (("panel-left-open", "Expand sidebar (Ctrl+B)") if collapsed
                     else ("panel-left-close", "Collapse sidebar (Ctrl+B)"))
        self.sidebar_button.setIcon(icon(name, COLORS["text_dim"], 18))
        self.sidebar_button.setToolTip(tip)

    def show_split_open(self, open_: bool) -> None:
        self.split_button.setChecked(open_)
        self.split_button.setToolTip("Close split view (Ctrl+\\)" if open_ else "Split view (Ctrl+\\)")

    def focus_search(self) -> None:
        self.search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.search.selectAll()

    def _go(self, text: str) -> None:
        """Open the page whose title is ``text`` (or the only one containing it)."""
        wanted = text.strip().lower()
        if not wanted:
            return
        exact = [page_id for title, page_id in self._titles.items() if title.lower() == wanted]
        partial = [page_id for title, page_id in self._titles.items() if wanted in title.lower()]
        matches = exact or partial
        if len(matches) == 1:
            self.page_requested.emit(matches[0])
            self.search.clear()
            self.search.clearFocus()
