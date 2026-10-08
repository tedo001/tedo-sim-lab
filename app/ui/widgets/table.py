"""A dense, read-only table with the lab's look."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ..theme import mono_font

__all__ = ["DataTable"]


class DataTable(QTableWidget):
    """Rows of text, with optional per-cell widgets (pills) and monospace columns.

    The table sizes itself to its rows (no inner scroll bar) so it sits inside a
    scrolling page like any other card content.
    """

    ROW_HEIGHT = 30

    def __init__(self, headers: Sequence[str], *, mono_columns: Sequence[int] = (),
                 stretch_column: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setObjectName("DataTable")
        self._mono_columns = set(mono_columns)
        self._widget_widths: dict[int, int] = {}  # Qt's size-to-content ignores cell widgets
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(self.ROW_HEIGHT)
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setHighlightSections(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if stretch_column is not None:
            header.setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)
        else:
            header.setStretchLastSection(True)

    def add_row(self, cells: Sequence[str | QWidget]) -> int:
        row = self.rowCount()
        self.insertRow(row)
        for column, value in enumerate(cells):
            if isinstance(value, QWidget):
                holder = QWidget()
                box = QHBoxLayout(holder)
                box.setContentsMargins(0, 0, 0, 0)  # the item padding already insets the cell
                box.addWidget(value)
                box.addStretch(1)
                self.setCellWidget(row, column, holder)
                value.ensurePolished()
                width = value.sizeHint().width() + 16  # + the 8 px item padding each side
                self._widget_widths[column] = max(self._widget_widths.get(column, 0), width)
            else:
                item = QTableWidgetItem(value)
                if column in self._mono_columns:
                    item.setFont(mono_font())
                self.setItem(row, column, item)
        self._fit_height()
        return row

    def sizeHintForColumn(self, column: int) -> int:
        return max(super().sizeHintForColumn(column), self._widget_widths.get(column, 0))

    def clear_rows(self) -> None:
        self.setRowCount(0)
        self._widget_widths.clear()
        self._fit_height()

    def showEvent(self, event: QShowEvent) -> None:
        self._fit_height()
        super().showEvent(event)

    def _fit_height(self) -> None:
        """Exactly tall enough for the header and every row (measured after styling)."""
        header = self.horizontalHeader()
        header.ensurePolished()
        height = header.sizeHint().height() + self.verticalHeader().length() + 2 * self.frameWidth()
        self.setFixedHeight(height)
