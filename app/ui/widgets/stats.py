"""Headline numbers: a strip of stat tiles (label, hero value, one-line note).

Values wear text colours, never a series or status colour; a status, when one
applies, is a pill beside the value so it never relies on colour alone.
"""

from __future__ import annotations

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QFrame, QGridLayout, QSizePolicy, QVBoxLayout, QWidget

from .basics import ElidedLabel, label

__all__ = ["StatStrip", "StatTile"]


class StatTile(QFrame):
    def __init__(self, title: str, value: str = "—", note: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatTile")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(120)
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
    """Tiles side by side in one bordered strip, divided by hairlines; they wrap onto more
    rows when the strip is too narrow (half-width split view)."""

    TILE_WIDTH = 170

    def __init__(self, tiles: dict[str, StatTile], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatStrip")
        self.tiles = tiles
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(0)
        self._columns = 0
        self._place(len(tiles))

    def columns(self) -> int:
        return self._columns

    def _place(self, columns: int) -> None:
        if columns == self._columns:
            return
        self._columns = columns
        for tile in self.tiles.values():
            self._grid.removeWidget(tile)
        for index, tile in enumerate(self.tiles.values()):
            row, column = divmod(index, columns)
            tile.setProperty("first", "true" if column == 0 else "false")
            tile.style().unpolish(tile)
            tile.style().polish(tile)
            self._grid.addWidget(tile, row, column)
        for column in range(len(self.tiles)):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
        self.updateGeometry()

    def resizeEvent(self, event: QResizeEvent) -> None:
        count = len(self.tiles)
        fits = min(count, max(1, event.size().width() // self.TILE_WIDTH))
        rows = -(-count // fits)
        self._place(-(-count // rows))  # balanced: 4 tiles in 3 places → 2 × 2, not 3 + 1
        super().resizeEvent(event)

    def __getitem__(self, key: str) -> StatTile:
        return self.tiles[key]
