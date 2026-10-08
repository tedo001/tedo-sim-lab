"""Page scaffolding: a scrolling page with a heading, and flat cards."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from ..theme.tokens import SIZES
from .basics import label

__all__ = ["Card", "Page", "PageHead"]


class PageHead(QWidget):
    """Title, a one-line monospace caption, and actions aligned right."""

    def __init__(self, title: str, caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = label(title, "PageTitle")
        self.caption = label(caption, "PageCaption", selectable=True)
        self.caption.setVisible(bool(caption))
        text.addWidget(self.title)
        text.addWidget(self.caption)
        row.addLayout(text, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)

    def set_caption(self, caption: str) -> None:
        self.caption.setText(caption)
        self.caption.setVisible(bool(caption))

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget, 0, Qt.AlignmentFlag.AlignBottom)


class Page(QWidget):
    """Base for every page: a heading over a scrolling body.

    Subclasses add content to :attr:`body` (a ``QVBoxLayout``).
    """

    def __init__(self, title: str, caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        margin = SIZES["page_margin"]
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.head = PageHead(title, caption)
        head_box = QVBoxLayout()
        head_box.setContentsMargins(margin, margin - 4, margin, SIZES["gap"])
        head_box.addWidget(self.head)
        outer.addLayout(head_box)

        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("PageBody")
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(margin, 0, margin, margin)
        self.body.setSpacing(SIZES["gap"])
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)


class Card(QFrame):
    """A flat panel: a head row (title, caption, one optional action) over content.

    Add content to :attr:`content`. ``padded=False`` lets a table run edge to edge.
    """

    def __init__(self, title: str = "", caption: str = "", *, padded: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.head = QFrame()
        self.head.setObjectName("CardHead")
        head_row = QHBoxLayout(self.head)
        head_row.setContentsMargins(14, 10, 10, 10)
        self.title = label(title, "CardTitle")
        self.caption = label(caption, "CardCaption")
        self.caption.setVisible(bool(caption))
        head_row.addWidget(self.title)
        head_row.addSpacing(6)
        head_row.addWidget(self.caption)
        head_row.addStretch(1)
        self._head_actions = head_row
        self.head.setVisible(bool(title))
        outer.addWidget(self.head)

        self.content = QVBoxLayout()
        pad = 14 if padded else 0
        self.content.setContentsMargins(pad, pad - 2 if padded else 0, pad, pad)
        self.content.setSpacing(8)
        outer.addLayout(self.content)

    def add_head_widget(self, widget: QWidget) -> None:
        self._head_actions.addWidget(widget)

    def add(self, item: QWidget | QLayout) -> None:
        if isinstance(item, QLayout):
            self.content.addLayout(item)
        else:
            self.content.addWidget(item)
