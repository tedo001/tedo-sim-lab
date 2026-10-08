"""The left navigation: sections from :data:`app.navigation.NAV`, one item per page.

Collapsible (Ctrl+B) to an icon rail; in the rail each item's tooltip names the page.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

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
        self._tag = label(tag, "NavTag") if tag else None
        row.addWidget(self._icon)
        row.addWidget(self._text, 1)
        if self._tag:
            row.addWidget(self._tag)
        self._row = row
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

    def set_compact(self, compact: bool) -> None:
        """Icon only (sidebar collapsed) or icon, title and tag."""
        self._text.setVisible(not compact)
        if self._tag:
            self._tag.setVisible(not compact)
        inset = (SIZES["sidebar_rail"] - SIZES["icon"]) // 2 - 2
        self._row.setContentsMargins(inset if compact else 12, 0, 0 if compact else 12, 0)
        self.setToolTip(f"{self.spec.title}: {self.spec.summary}" if compact else self.spec.summary)


class Sidebar(QFrame):
    page_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(SIZES["sidebar_width"])
        self.items: dict[str, NavItem] = {}
        self._headings: list[QLabel] = []
        self._dividers: list[QFrame] = []
        self._collapsed = False

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
                heading_label = label(heading.upper(), "NavSection")
                divider = QFrame()
                divider.setObjectName("NavDivider")
                divider.setFixedHeight(1)
                divider.hide()
                self._headings.append(heading_label)
                self._dividers.append(divider)
                column.addWidget(heading_label)
                column.addWidget(divider)
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

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setFixedWidth(SIZES["sidebar_rail"] if collapsed else SIZES["sidebar_width"])
        for heading in self._headings:
            heading.setVisible(not collapsed)
        for divider in self._dividers:
            divider.setVisible(collapsed)
        for item in self.items.values():
            item.set_compact(collapsed)

    def collapsed(self) -> bool:
        return self._collapsed
