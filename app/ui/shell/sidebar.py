"""The left navigation: sections from :data:`app.navigation.NAV`, one item per page."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ...navigation import SECTIONS, PageSpec, pages_in
from ..icons import pixmap
from ..theme.tokens import COLORS, SIZES
from ..widgets import label, repolish

__all__ = ["NavItem", "Sidebar"]


def _tag(spec: PageSpec) -> str:
    """Faint marker for pages that arrive in a later release; V0.1 pages get none."""
    if spec.planned_for in (None, "v0.1"):
        return ""
    return "TBD" if spec.planned_for == "TBD" else spec.planned_for


class NavItem(QPushButton):
    """Icon, title and an optional release tag; ``active`` marks the current page."""

    def __init__(self, spec: PageSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("NavItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)
        self.setToolTip(spec.summary)
        self.setAccessibleName(spec.title)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(10)
        self._icon = QLabel()
        self._text = label(spec.title, "NavText")
        tag = _tag(spec)
        row.addWidget(self._icon)
        row.addWidget(self._text, 1)
        if tag:
            row.addWidget(label(tag, "NavTag"))
        for child in self.findChildren(QLabel):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        colour = COLORS["text"] if active else COLORS["text_dim"]
        self._icon.setPixmap(pixmap(self.spec.icon, colour, SIZES["icon"]))
        for widget in (self, self._text):
            widget.setProperty("active", "true" if active else "false")
            repolish(widget)

    def is_active(self) -> bool:
        return self.property("active") == "true"


class Sidebar(QFrame):
    page_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(SIZES["sidebar_width"])
        self.items: dict[str, NavItem] = {}

        body = QWidget()
        body.setObjectName("SidebarBody")
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 6, 0, 8)
        column.setSpacing(0)
        for section, heading in SECTIONS:
            specs = [spec for spec in pages_in(section) if spec.in_sidebar]
            if not specs:
                continue
            if heading:
                column.addWidget(label(heading.upper(), "NavSection"))
            for spec in specs:
                item = NavItem(spec)
                item.clicked.connect(lambda _=False, page_id=spec.id: self.page_requested.emit(page_id))
                self.items[spec.id] = item
                column.addWidget(item)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_current(self, page_id: str) -> None:
        for item_id, item in self.items.items():
            item.set_active(item_id == page_id)
