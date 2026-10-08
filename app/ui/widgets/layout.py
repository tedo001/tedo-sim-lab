"""Page scaffolding: a scrolling page with a heading, and flat cards."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme.tokens import SIZES
from .basics import label

__all__ = ["Card", "Page", "PageHead", "ResponsiveRow"]


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

    def sizeHint(self) -> QSize:
        """Tall enough for wrapped text at the current width: the vertical policy is Maximum, so
        a hint computed for a wider card would clip lines when the card is narrow."""
        hint = super().sizeHint()
        if self.width() > 0 and self.hasHeightForWidth():
            hint.setHeight(max(hint.height(), self.heightForWidth(self.width())))
        return hint

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()

    def add(self, item: QWidget | QLayout) -> None:
        if isinstance(item, QLayout):
            self.content.addLayout(item)
        else:
            self.content.addWidget(item)


class ResponsiveRow(QWidget):
    """Cards side by side, stacked instead when the row is narrower than ``breakpoint``.

    Keeps pages usable at half width (split view) without sideways scrolling.
    """

    def __init__(self, widgets: list[tuple[QWidget, int]], *, breakpoint: int = 900,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.breakpoint = breakpoint
        self._stretches = [stretch for _, stretch in widgets]
        self._box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(SIZES["gap"])
        for widget, stretch in widgets:
            self._box.addWidget(widget, stretch, Qt.AlignmentFlag.AlignTop)

    def stacked(self) -> bool:
        return self._box.direction() == QBoxLayout.Direction.TopToBottom

    def _widgets(self) -> list[QWidget]:
        items = (self._box.itemAt(i) for i in range(self._box.count()))
        return [item.widget() for item in items if item is not None and item.widget() is not None]

    def _side_by_side_width(self) -> int:
        widgets = self._widgets()
        return (sum(w.minimumSizeHint().width() for w in widgets)
                + self._box.spacing() * max(len(widgets) - 1, 0))

    def minimumSizeHint(self) -> QSize:
        """As narrow as the widest child: below that the row stacks instead of overflowing."""
        hint = super().minimumSizeHint()
        widest = max((w.minimumSizeHint().width() for w in self._widgets()), default=0)
        return QSize(widest, hint.height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        width = event.size().width()
        narrow = width < self.breakpoint or width < self._side_by_side_width()
        if narrow != self.stacked():
            self._box.setDirection(QBoxLayout.Direction.TopToBottom if narrow
                                   else QBoxLayout.Direction.LeftToRight)
            for index, stretch in enumerate(self._stretches):
                self._box.setStretch(index, 0 if narrow else stretch)
            self.updateGeometry()  # the row's height changed: ask the page to lay out again
        super().resizeEvent(event)
