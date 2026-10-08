"""The title row: wordmark, version, Ctrl+K page search and settings.

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


class TopBar(QFrame):
    page_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(SIZES["topbar_height"])
        self._titles = {spec.title: spec.id for spec in NAV}

        row = QHBoxLayout(self)
        row.setContentsMargins(16, 0, 12, 0)
        row.setSpacing(10)
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

        self.settings_button = QToolButton()
        self.settings_button.setObjectName("TopBarButton")
        self.settings_button.setIcon(icon("settings", COLORS["text_dim"], 18))
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setToolTip("Settings")
        self.settings_button.setAccessibleName("Settings")
        self.settings_button.clicked.connect(lambda: self.page_requested.emit("settings"))
        row.addWidget(self.settings_button)

        search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        search_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        search_shortcut.activated.connect(self.focus_search)

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
