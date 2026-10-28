"""The Dataset Hub's catalogue: every dataset card (images, audio, text, tables, synthetic sets,
image–text pairs; shipped, project and imported) with its licence, source, size, classes and access
requirements; the download controls appear only for datasets the lab can load and whose terms let
it fetch them. A downloaded dataset shows a few examples, opens in the lab for its kind of data and
starts an experiment in the Experiment Builder."""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Modality
from labs.common.explore import explorable

from ...navigation import page
from ...services.context import AppContext
from ...services.starters import lab_for, starter_spec, trainable_tasks
from ..icons import icon
from ..widgets import Card, DataTable, FlowGrid, KeyValues, label
from .builder.choosers import DatasetDownload
from .builder.form import task_title
from .catalog_common import FilterRow, licence_pill, status_pill
from .dataset_samples import DatasetSamples

__all__ = ["DatasetCatalog"]


def _splits(splits: dict[str, int]) -> str:
    return " · ".join(f"{name} {count:,}" for name, count in splits.items()) or "not stated"


class DatasetCatalog(QWidget):
    #: Draw this dataset's rows in 3-D (a card id).
    explore_requested = Signal(str)

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
        self.source_button = QPushButton(icon("external-link"), "Open source page")
        self.source_button.clicked.connect(lambda: self._open("source"))
        self.terms_button = QPushButton(icon("scale"), "Read the terms")
        self.terms_button.clicked.connect(lambda: self._open("terms"))
        self.explore_button = QPushButton(icon("rotate-3d"), "Explore in 3-D")
        self.explore_button.clicked.connect(
            lambda: self.selected and self.explore_requested.emit(self.selected))
        self.lab_button = QPushButton(icon("flask-conical"), "Open lab")
        self.lab_button.clicked.connect(self._open_lab)
        self.builder_button = QPushButton(icon("sliders-horizontal"), "Use in Experiment Builder")
        self.builder_button.setObjectName("Primary")
        self.builder_button.clicked.connect(self._use_in_builder)
        self.builder_note = label("", "CardCaption", wrap=True)
        row = [self.builder_button, self.lab_button, self.explore_button, self.source_button,
               self.terms_button]
        for widget in (self.description, self.values, self.notes, self.citation, self.loader, self.download):
            self.detail.add(widget)
        self.detail.add(FlowGrid(row, 210, fewest=2))
        self.detail.add(self.builder_note)
        layout.addWidget(self.detail)
        self.samples = DatasetSamples(ctx)
        layout.addWidget(self.samples)
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
        self._show_explore(card.id)
        self._show_next(card, status)
        self.samples.set_card(card.id if status == "ready" else None)

    def _show_explore(self, card_id: str) -> None:
        """Rows can be drawn in 3-D for tables and MNIST-style image sets on this computer."""
        registry = self.ctx.catalog.datasets
        try:
            adapter = registry.adapter(card_id)
        except (ImportError, AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            adapter = None
        if adapter is None or not explorable(adapter):
            self.explore_button.setVisible(False)
            return
        here = adapter.local_status(registry.root) == "present"
        self.explore_button.setVisible(True)
        self.explore_button.setEnabled(here)
        self.explore_button.setToolTip("Its rows as points in 3-D, coloured by class" if here
                                       else "Download it first")

    def _stable_tasks(self) -> set:
        return {task for task, state in self.ctx.catalog.runners.task_maturity().items() if state == "stable"}

    def _show_next(self, card, status: str) -> None:  # noqa: ANN001 (a DatasetCard)
        """Open lab, and Use in Experiment Builder for datasets a stable runner trains."""
        lab = lab_for(card)
        self.lab_button.setText(f"Open {page(lab).title}" if lab != "dataset_hub" else "Open lab")
        self.lab_button.setVisible(lab != "dataset_hub")
        tasks = trainable_tasks(card, self._stable_tasks())
        self.builder_button.setVisible(bool(tasks))
        ready = status == "ready"
        self.builder_button.setEnabled(ready)
        self.builder_button.setToolTip("" if ready else "Download it first")
        self.builder_note.setText(
            f"Trains as {', '.join(task_title(t).lower() for t in tasks)}." if tasks else
            "No model in this release trains on it yet; it is listed with its licence and source." if
            status != "catalog_only" else "")
        self.builder_note.setVisible(bool(self.builder_note.text()))

    def _open_lab(self) -> None:
        if self.selected is not None:
            self.ctx.navigate(lab_for(self.ctx.catalog.datasets.get(self.selected)))

    def _use_in_builder(self) -> None:
        if self.selected is None:
            return
        card = self.ctx.catalog.datasets.get(self.selected)
        tasks = trainable_tasks(card, self._stable_tasks())
        try:
            spec = starter_spec(card, tasks[0], self.ctx.catalog.models)
        except (IndexError, LookupError, ValueError) as exc:
            self.builder_note.setText(str(exc).strip("'\""))
            return
        self.ctx.experiments.open_in_builder(spec, self.ctx.navigate)

    def _open(self, which: str) -> None:
        if self.selected is None:
            return
        card = self.ctx.catalog.datasets.get(self.selected)
        url = card.source_url if which == "source" else card.license.url
        if url:
            QDesktopServices.openUrl(QUrl(url))
