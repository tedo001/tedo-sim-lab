"""The Plugin Store's Browse card for dataset and model sources (Kaggle, Hugging Face,
Roboflow): search, see each item's licence and access terms, download on request after
confirming the licence. Restricted items show their page and why, never a download button."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QPushButton, QWidget

from labs.common.remote import RemoteItem

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Pill, label

__all__ = ["BrowseCard"]


class BrowseCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Browse", "search, read the terms, download into this workspace", parent=parent)
        self.ctx = ctx
        self.name = ""
        self.items: list[RemoteItem] = []
        self.downloaded: Path | None = None
        self._search_job: str | None = None
        self._download_job: str | None = None
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search")
        self.query.setClearButtonEnabled(True)
        self.query.returnPressed.connect(self.search)
        row.addWidget(self.query, 1)
        self.search_button = QPushButton(icon("search"), "Search")
        self.search_button.clicked.connect(self.search)
        row.addWidget(self.search_button)
        self.add(row)
        self.state = label("", "CardCaption", wrap=True)
        self.add(self.state)
        self.table = DataTable(("Name", "Kind", "Licence", "Size"), mono_columns=(3,), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selected)
        self.add(self.table)

        self.facts = KeyValues([("Item", "—"), ("Licence", "—"), ("Page", "—"), ("Saved to", "—")])
        self.add(self.facts)
        self.restriction = label("", "Problem", wrap=True)
        self.add(self.restriction)
        self.accept = QCheckBox("I have read this item's licence and will use it under its terms")
        self.accept.toggled.connect(lambda _: self._update_buttons())
        self.add(self.accept)
        actions = QHBoxLayout()
        self.page_button = QPushButton(icon("external-link"), "Open its page")
        self.page_button.clicked.connect(self._open_page)
        self.download_button = QPushButton(icon("download"), "Download")
        self.download_button.setObjectName("Primary")
        self.download_button.clicked.connect(self.download)
        self.folder_button = QPushButton(icon("folder-open"), "Open folder")
        self.folder_button.clicked.connect(self._open_folder)
        for button in (self.page_button, self.download_button, self.folder_button):
            actions.addWidget(button)
        actions.addStretch(1)
        self.add(actions)
        self.progress = label("", "Mono", wrap=True)
        self.add(self.progress)
        ctx.plugins.finished.connect(self._finished)
        ctx.jobs.job_progress.connect(self._progress)

    # Showing ----------------------------------------------------------------------------------
    def set_plugin(self, name: str) -> None:
        self.name = name
        self.items = []
        self.table.clear_rows()
        manifest = self.ctx.plugins.plugin(name).manifest
        self.set_title(f"Browse {manifest.title}")
        self.query.setPlaceholderText("Filter your projects" if name == "roboflow" else
                                      "Search by keyword")
        self.state.setText("")
        self._show(None)

    def current(self) -> RemoteItem | None:
        rows = self.table.selectionModel().selectedRows()
        return self.items[rows[0].row()] if rows and rows[0].row() < len(self.items) else None

    def _selected(self) -> None:
        self._show(self.current())

    def _show(self, item: RemoteItem | None) -> None:
        self.facts.set_value("Item", f"{item.id} · {item.kind}" if item else "—")
        self.facts.set_value("Licence", item.licence_text if item else "—")
        self.facts.set_value("Page", item.url if item else "—")
        self.facts.set_value("Saved to", str(self.ctx.plugins.plugin(self.name).destination(
            item, self.ctx.paths.workspace)) if item and self.name else "—")
        self.restriction.setText(f"Not downloaded by the lab: {item.restriction}" if item and item.restriction
                                 else "")
        self.restriction.setVisible(bool(self.restriction.text()))
        self.accept.setChecked(False)
        self.accept.setText(f"I have read the licence ({item.licence_text}) and will use it under its terms"
                            if item else "I have read this item's licence and will use it under its terms")
        self._update_buttons()

    def _update_buttons(self) -> None:
        item = self.current()
        downloadable = item is not None and not item.restriction
        idle = self._download_job is None
        self.accept.setEnabled(downloadable and idle)
        self.page_button.setEnabled(item is not None)
        self.download_button.setEnabled(downloadable and self.accept.isChecked() and idle)
        self.download_button.setToolTip("" if downloadable or item is None else item.restriction)
        self.folder_button.setEnabled(self.downloaded is not None)
        self.search_button.setEnabled(self._search_job is None and bool(self.name))

    # Acting ------------------------------------------------------------------------------------
    def search(self) -> str | None:
        if not self.name or self._search_job is not None:
            return None
        self.state.setText("Searching…")
        self._search_job = self.ctx.plugins.search(self.name, self.query.text())
        self._update_buttons()
        return self._search_job

    def download(self) -> str | None:
        item = self.current()
        if item is None or item.restriction or not self.accept.isChecked() or self._download_job:
            return None
        self.progress.setText(f"Starting {item.title}…")
        self._download_job = self.ctx.plugins.download(self.name, item)
        self._update_buttons()
        return self._download_job

    def _progress(self, job_id: str, fraction: float, message: str) -> None:
        if job_id == self._download_job:
            share = f"{fraction:.0%} · " if fraction >= 0 else ""
            self.progress.setText(share + message)

    def _finished(self, job_id: str, name: str, action: str, result: object, error: str) -> None:
        if job_id == self._search_job:
            self._search_job = None
            if name == self.name:
                self._fill(result if not error else [], error)
        elif job_id == self._download_job:
            self._download_job = None
            if error:
                self.progress.setText(f"Download failed: {error}")
            else:
                self.downloaded = Path(str(result))
                self.progress.setText(f"Saved to {self.downloaded} (the licence is in source.json there). "
                                      "CSV files can be imported in the Classical ML lab.")
        self._update_buttons()

    def _fill(self, items: object, error: str) -> None:
        self.items = list(items) if isinstance(items, list) else []
        self.table.clear_rows()
        for item in self.items:
            licence = Pill(item.licence or "Unspecified", "fail" if item.restriction else
                           ("ok" if item.licence else "planned"))
            licence.setToolTip(item.restriction or item.licence_text)
            self.table.add_row([item.title, item.kind, licence, item.size_text])
        self.state.setText(error or (f"{len(self.items)} found." if self.items else "Nothing found."))
        self._show(None)

    def _open_page(self) -> None:
        item = self.current()
        if item:
            QDesktopServices.openUrl(QUrl(item.url))

    def _open_folder(self) -> None:
        if self.downloaded:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.downloaded)))
