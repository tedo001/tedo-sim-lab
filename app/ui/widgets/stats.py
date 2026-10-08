"""Headline numbers: a strip of stat tiles (label, hero value, one-line note).

Values wear text colours, never a series or status colour; a status, when one
applies, is a pill beside the value so it never relies on colour alone.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from .basics import ElidedLabel, label

__all__ = ["StatStrip", "StatTile"]


class StatTile(QFrame):
    def __init__(self, title: str, value: str = "—", note: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatTile")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 12, 16, 12)
        column.setSpacing(2)
        self.title = label(title, "StatLabel")
        self.value = label(value, "StatValue")
        self.note = ElidedLabel(note, "StatNote")
        self.value.setAccessibleName(title)
        column.addWidget(self.title)
        column.addWidget(self.value)
        column.addWidget(self.note)

    def set(self, value: str, note: str | None = None) -> None:
        self.value.setText(value)
        if note is not None and note != self.note.full_text:
            self.note.set_full_text(note)


class StatStrip(QFrame):
    """Tiles side by side in one bordered strip, divided by hairlines."""

    def __init__(self, tiles: dict[str, StatTile], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatStrip")
        self.tiles = tiles
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for index, tile in enumerate(tiles.values()):
            tile.setProperty("first", "true" if index == 0 else "false")
            row.addWidget(tile, 1)

    def __getitem__(self, key: str) -> StatTile:
        return self.tiles[key]
