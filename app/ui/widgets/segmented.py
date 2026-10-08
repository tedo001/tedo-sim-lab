"""A row of mutually exclusive buttons that switches a stack of views (a flat tab bar)."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

__all__ = ["Segmented"]


class Segmented(QWidget):
    """``items`` are (key, title); ``changed`` sends the chosen key."""

    changed = Signal(str)

    def __init__(self, items: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for index, (key, title) in enumerate(items):
            button = QPushButton(title)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setProperty("edge", "first" if index == 0 else "last" if index == len(items) - 1 else "")
            button.clicked.connect(lambda _=False, k=key: self.select(k))
            self.group.addButton(button)
            self.buttons[key] = button
            row.addWidget(button)
        row.addStretch(1)
        if items:
            self.buttons[items[0][0]].setChecked(True)

    @property
    def current(self) -> str:
        return next(key for key, button in self.buttons.items() if button.isChecked())

    def select(self, key: str) -> None:
        self.buttons[key].setChecked(True)
        self.changed.emit(key)
