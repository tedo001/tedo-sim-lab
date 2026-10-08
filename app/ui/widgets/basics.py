"""Small building blocks: text labels, pills and key/value rows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QGuiApplication, QResizeEvent
from PyQt6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

__all__ = ["KeyValues", "PathLabel", "Pill", "label", "repolish"]


def repolish(widget: QWidget) -> None:
    """Re-apply the style sheet after a dynamic property changed."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


def label(text: str, name: str = "Body", *, wrap: bool = False, selectable: bool = False) -> QLabel:
    """A ``QLabel`` styled by its object name (``Body``, ``Mono``, ``CardTitle`` ...)."""
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


class Pill(QLabel):
    """A short status or licence tag, tinted by ``tone``.

    Tones: ``ok``, ``warn``, ``fail``, ``info``, ``experimental``, ``planned``.
    """

    def __init__(self, text: str, tone: str = "planned", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Pill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        self.setProperty("tone", tone)
        repolish(self)

    def tone(self) -> str:
        return str(self.property("tone"))


class PathLabel(QLabel):
    """A path in the monospace face, shortened in the middle when it does not fit.

    The full path is in the tooltip and behind a right-click "Copy path".
    """

    def __init__(self, path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("KvValueMono")
        self.full_text = str(path)
        self.setToolTip(self.full_text)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        copy = QAction("Copy path", self)
        copy.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.full_text))
        self.addAction(copy)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.setText(self.full_text)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        shown = self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideMiddle,
                                              max(self.width(), 0))
        if shown != self.text():
            self.setText(shown)


class KeyValues(QWidget):
    """Aligned ``key  value`` rows, values optionally in the monospace face.

    Machine-written values (paths, URIs, versions, hashes) go in mono; words
    people wrote stay in the text face. A :class:`~pathlib.Path` value becomes a
    :class:`PathLabel`.
    """

    def __init__(self, rows: Iterable[tuple[str, str | Path | QWidget]] = (), *, mono: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mono = mono
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(7)
        self._grid.setColumnStretch(1, 1)
        for key, value in rows:
            self.add_row(key, value)

    def add_row(self, key: str, value: str | Path | QWidget) -> None:
        row = self._grid.rowCount()
        key_label = label(key, "KvKey")
        key_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._grid.addWidget(key_label, row, 0)
        if isinstance(value, Path):
            value = PathLabel(value)
        elif isinstance(value, str):
            value = label(value, "KvValueMono" if self._mono else "KvValue",
                          wrap=True, selectable=True)
        self._grid.addWidget(value, row, 1)

    def value_text(self, key: str) -> str | None:
        """The text shown for ``key`` (tests and accessibility)."""
        for row in range(self._grid.rowCount()):
            key_item = self._grid.itemAtPosition(row, 0)
            value_item = self._grid.itemAtPosition(row, 1)
            if key_item and isinstance(key_item.widget(), QLabel) and \
                    key_item.widget().text() == key and value_item:
                widget = value_item.widget()
                if isinstance(widget, PathLabel):
                    return widget.full_text
                return widget.text() if isinstance(widget, QLabel) else None
        return None
