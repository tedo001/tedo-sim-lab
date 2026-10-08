"""Small building blocks: text labels, pills and key/value rows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QResizeEvent
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

__all__ = ["ElidedLabel", "KeyValues", "PathLabel", "Pill", "label", "repolish"]


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


class ElidedLabel(QLabel):
    """One line that shortens with "…" when it does not fit, so narrow panes never clip.

    The full text is in the tooltip and behind a right-click "Copy".
    """

    def __init__(self, text: str, name: str = "KvValueMono", *,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight, copy_label: str = "Copy",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self._mode = mode
        self.full_text = text
        self.setToolTip(text)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        copy = QAction(copy_label, self)
        copy.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.full_text))
        self.addAction(copy)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.setText(text)

    def set_full_text(self, text: str) -> None:
        self.full_text = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self.full_text, self._mode, max(self.width(), 0))
        if shown != self.text():
            self.setText(shown)


class PathLabel(ElidedLabel):
    """A path in the monospace face, shortened in the middle (both ends stay readable)."""

    def __init__(self, path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(str(path), mode=Qt.TextElideMode.ElideMiddle, copy_label="Copy path",
                         parent=parent)


class KeyValues(QWidget):
    """Aligned ``key  value`` rows, values optionally in the monospace face.

    Machine-written values (paths, URIs, versions, hashes) go in mono; words
    people wrote stay in the text face. Every value is one line that shortens
    with "…" when space runs out (full text in the tooltip); a :class:`~pathlib.Path`
    shortens in the middle.
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
            value = ElidedLabel(value, "KvValueMono" if self._mono else "KvValue")
        self._grid.addWidget(value, row, 1)

    def set_value(self, key: str, value: str) -> None:
        """Replace the text shown for ``key`` (adds the row if it is new)."""
        for row in range(self._grid.rowCount()):
            key_item = self._grid.itemAtPosition(row, 0)
            value_item = self._grid.itemAtPosition(row, 1)
            if key_item and key_item.widget().text() == key and value_item:
                widget = value_item.widget()
                if isinstance(widget, ElidedLabel):
                    widget.set_full_text(value)
                    return
        self.add_row(key, value)

    def value_text(self, key: str) -> str | None:
        """The text shown for ``key`` (tests and accessibility)."""
        for row in range(self._grid.rowCount()):
            key_item = self._grid.itemAtPosition(row, 0)
            value_item = self._grid.itemAtPosition(row, 1)
            if key_item and isinstance(key_item.widget(), QLabel) and \
                    key_item.widget().text() == key and value_item:
                widget = value_item.widget()
                if isinstance(widget, ElidedLabel):
                    return widget.full_text
                return widget.text() if isinstance(widget, QLabel) else None
        return None
