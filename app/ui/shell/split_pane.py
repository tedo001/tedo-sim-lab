"""The second pane of split view: any page, side by side with the main one.

Pages here are separate instances from the main stack (a widget lives in one
place only), built on first use. Both copies read the same context, so live
data (jobs, resources) updates in both.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...navigation import NAV, page
from ..icons import icon
from ..theme.tokens import COLORS

__all__ = ["SplitPane"]


class SplitPane(QFrame):
    page_changed = Signal(str)
    close_requested = Signal()

    def __init__(self, build: Callable[[str], QWidget], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SplitPane")
        self._build = build
        self._pages: dict[str, QWidget] = {}
        self._current = ""

        head = QFrame()
        head.setObjectName("SplitHead")
        row = QHBoxLayout(head)
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(6)
        self.chooser = QComboBox()
        self.chooser.setObjectName("SplitChooser")
        self.chooser.setAccessibleName("Page in split view")
        for spec in NAV:
            self.chooser.addItem(icon(spec.icon, COLORS["text_dim"], 14), spec.title, spec.id)
        self.chooser.currentIndexChanged.connect(
            lambda index: self.set_page(self.chooser.itemData(index)))
        row.addWidget(self.chooser, 1)
        self.close_button = QToolButton()
        self.close_button.setObjectName("TopBarButton")
        self.close_button.setIcon(icon("x", COLORS["text_dim"], 16))
        self.close_button.setIconSize(QSize(16, 16))
        self.close_button.setToolTip("Close split view (Ctrl+\\)")
        self.close_button.setAccessibleName("Close split view")
        self.close_button.clicked.connect(self.close_requested.emit)
        row.addWidget(self.close_button)

        self.stack = QStackedWidget()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(head)
        column.addWidget(self.stack, 1)

    def set_page(self, page_id: str) -> None:
        page(page_id)  # KeyError for an unknown id
        if page_id not in self._pages:
            widget = self._build(page_id)
            widget.setProperty("pageId", page_id)
            self._pages[page_id] = widget
            self.stack.addWidget(widget)
        self.stack.setCurrentWidget(self._pages[page_id])
        index = self.chooser.findData(page_id)
        if index != self.chooser.currentIndex():
            self.chooser.blockSignals(True)
            self.chooser.setCurrentIndex(index)
            self.chooser.blockSignals(False)
        if page_id != self._current:
            self._current = page_id
            self.page_changed.emit(page_id)

    def current_page_id(self) -> str:
        return self._current
