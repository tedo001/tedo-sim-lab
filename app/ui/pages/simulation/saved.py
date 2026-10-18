"""Saved simulations: every one kept in ``results/simulations``, newest first — open one again in
its tool, show its folder, or delete it."""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from labs.simulation import records

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, DataTable, Pill, label

__all__ = ["SavedView"]

_TONES = {"circuit": "info", "waves": "experimental", "image": "ok", "scenes": "warn", "camera": "planned"}


class SavedView(QWidget):
    open_requested = Signal(str, dict)   # kind, settings
    message = Signal(str)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.items: list[records.SavedSimulation] = []
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        card = Card("Saved simulations", "settings to run them again, and their outputs")
        self.caption = label("", "CardCaption", wrap=True)
        card.add(self.caption)
        self.table = DataTable(("Kind", "Name", "Saved", "Folder"), mono_columns=(2, 3), stretch_column=1)
        self.table.cellDoubleClicked.connect(lambda row, _c: self.open_row(row))
        card.add(self.table)
        row = QHBoxLayout()
        for text, name, slot in (("Open", "folder-open", lambda: self.open_row(self._row())),
                                 ("Show folder", "external-link", self.show_folder),
                                 ("Delete", "trash-2", lambda: self.delete_row(self._row()))):
            button = QPushButton(icon(name), text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        card.add(row)
        box.addWidget(card)
        box.addStretch(1)

    def refresh(self) -> None:
        self.items = records.list_saved(self.ctx.paths.results)
        self.table.clear_rows()
        for item in self.items:
            self.table.add_row([Pill(item.title, _TONES.get(item.kind, "planned")), item.name,
                                item.created.replace("T", " "), item.folder.name])
        folder = records.simulations_folder(self.ctx.paths.results)
        self.caption.setText(f"{len(self.items)} saved in {folder}" if self.items else
                             "Nothing saved yet: use Save on any tool.")

    def _row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def open_row(self, row: int) -> None:
        if 0 <= row < len(self.items):
            try:
                record = records.load(self.items[row].folder)
            except (OSError, ValueError) as exc:
                self.message.emit(str(exc))
                return
            self.open_requested.emit(record["kind"], record["settings"])

    def show_folder(self) -> None:
        row = self._row()
        folder = self.items[row].folder if 0 <= row < len(self.items) else \
            records.simulations_folder(self.ctx.paths.results)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def delete_row(self, row: int, *, confirm: bool = True) -> None:
        if not 0 <= row < len(self.items):
            return
        item = self.items[row]
        if confirm and QMessageBox.question(self, "Delete", f"Delete the saved simulation “{item.name}”?") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            records.delete(item, self.ctx.paths.results)
        except (OSError, ValueError) as exc:
            self.message.emit(str(exc))
            return
        self.message.emit(f"Deleted {item.name}")
        self.refresh()
