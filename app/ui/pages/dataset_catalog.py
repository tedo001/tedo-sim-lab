"""The Dataset Hub's catalogue: every dataset card (shipped, project and imported) with its
licence, source, size, classes and access requirements; the download controls appear only
for datasets the lab can load and whose terms let it fetch them."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Modality

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, label
from .builder.choosers import DatasetDownload
from .builder.form import task_title
from .catalog_common import FilterRow, licence_pill, status_pill

__all__ = ["DatasetCatalog"]


def _splits(splits: dict[str, int]) -> str:
    return " · ".join(f"{name} {count:,}" for name, count in splits.items()) or "not stated"


class DatasetCatalog(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.ids: list[str] = []
        self.selected: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.filters = FilterRow("Search names, tasks, tags…", [("Any kind", None)] + [
            (modality.value.capitalize(), modality) for modality in Modality])
        self.filters.connect(self.refresh)
        layout.addWidget(self.filters)
        listing = Card("Datasets", padded=False)
        self.table = DataTable(("Dataset", "Kind", "Size", "Licence", "Status"), mono_columns=(2,),
                               stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        listing.add(self.table)
        self.count = label("", "CardCaption")
        listing.add_head_widget(self.count)
        layout.addWidget(listing)

        self.detail = Card("")
        self.head_pills = QHBoxLayout()
        holder = QWidget()
        holder.setLayout(self.head_pills)
        self.head_pills.setContentsMargins(0, 0, 0, 0)
        self.detail.add_head_widget(holder)
        self.description = label("", "Body", wrap=True)
        self.values = KeyValues((("Source", ""), ("Licence", ""), ("Tasks", ""), ("Size", ""),
                                 ("Classes", ""), ("Splits", ""), ("Access", "")))
        self.notes = label("", "Body", wrap=True)
        self.citation = label("", "CardCaption", wrap=True, selectable=True)
        self.loader = label("", "CardCaption", wrap=True)
        self.download = DatasetDownload(ctx, show_status=False)
        buttons = QHBoxLayout()
        self.source_button = QPushButton(icon("external-link"), "Open source page")
        self.source_button.clicked.connect(lambda: self._open("source"))
        self.terms_button = QPushButton(icon("scale"), "Read the terms")
        self.terms_button.clicked.connect(lambda: self._open("terms"))
        buttons.addWidget(self.source_button)
        buttons.addWidget(self.terms_button)
        buttons.addStretch(1)
        for widget in (self.description, self.values, self.notes, self.citation, self.loader, self.download):
            self.detail.add(widget)
        self.detail.add(buttons)
        layout.addWidget(self.detail)
        ctx.downloads.finished.connect(lambda *_: self.refresh())
        ctx.downloads.imported.connect(lambda *_: self.refresh())
        self.refresh()

    # Listing ----------------------------------------------------------------------
    def refresh(self) -> None:
        registry = self.ctx.catalog.datasets
        cards = [card for card in registry.all()
                 if self.filters.matches(card.search_text(), card.modality, card.license.category)]
        self.ids = [card.id for card in cards]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for card in cards:
            self.table.add_row((card.name, card.modality.value, card.size or "—", licence_pill(card.license),
                                status_pill(registry.status(card.id))))
        self.table.blockSignals(False)
        self.count.setText(f"{len(cards)} of {len(registry)}")
        if self.selected not in self.ids:
            self.selected = self.ids[0] if self.ids else None
        if self.selected is not None:
            self.table.blockSignals(True)
            self.table.selectRow(self.ids.index(self.selected))
            self.table.blockSignals(False)
        self._show()

    def select(self, card_id: str) -> None:
        self.filters.reset()
        self.selected = card_id
        self.refresh()

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.ids[rows[0].row()]
            self._show()

    # Detail -----------------------------------------------------------------------
    def _show(self) -> None:
        self.detail.setVisible(self.selected is not None)
        if self.selected is None:
            return
        registry = self.ctx.catalog.datasets
        card = registry.get(self.selected)
        status, detail = registry.status_detail(card.id)
        self.detail.set_title(card.name)
        while self.head_pills.count():
            widget = self.head_pills.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.head_pills.addWidget(licence_pill(card.license))
        self.head_pills.addWidget(status_pill(status))
        self.description.setText(card.description)
        self.values.set_value("Source", card.source_url or "—")
        self.values.set_value("Licence", card.license.name)
        self.values.set_value("Tasks", ", ".join(task_title(task) for task in card.tasks) or "—")
        self.values.set_value("Size", card.size or "—")
        classes = f"{card.num_classes:,}" if card.num_classes else "—"
        if card.class_names:
            classes += f": {', '.join(card.class_names[:12])}" + ("…" if len(card.class_names) > 12 else "")
        self.values.set_value("Classes", classes)
        self.values.set_value("Splits", _splits(card.splits))
        self.values.set_value("Access", card.access_requirements or "No account needed")
        self.notes.setText(card.license.notes)
        self.notes.setVisible(bool(card.license.notes))
        self.citation.setText(f"Cite: {card.citation}" if card.citation else "")
        self.citation.setVisible(bool(card.citation))
        loadable = status in ("ready", "not_downloaded", "experimental")
        self.loader.setText("" if loadable else f"{detail} Use the source page to get it.")
        self.loader.setVisible(not loadable)
        self.download.setVisible(loadable)
        if loadable:
            self.download.set_card(card.id)
        self.source_button.setEnabled(bool(card.source_url))
        self.terms_button.setVisible(bool(card.license.url))

    def _open(self, which: str) -> None:
        if self.selected is None:
            return
        card = self.ctx.catalog.datasets.get(self.selected)
        url = card.source_url if which == "source" else card.license.url
        if url:
            QDesktopServices.openUrl(QUrl(url))
