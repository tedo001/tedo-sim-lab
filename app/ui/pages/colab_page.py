"""Google Colab: write an experiment as a Colab notebook (it trains with the lab's own worker
on Colab's GPU and zips the run folder), then import that zip back as a run of this workspace.
The lab never signs in to Google: you upload the notebook and download the zip yourself."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QPushButton, QWidget

from core.experiment_engine.spec import ExperimentSpec, SpecError, load_spec_text

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, Page, label

__all__ = ["ColabPage"]

COLAB_URL = "https://colab.research.google.com/"
STEPS = ("1. Write the notebook below.  2. On Colab: File → Upload notebook, pick it, choose a GPU "
         "runtime, run every cell (read the dataset's terms where the notebook asks).  3. The last cell "
         "downloads a zip; import it here. Colab's terms of service and usage limits apply.")


class ColabPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Google Colab", "train on Colab's GPU, keep the results here", parent)
        self.ctx = ctx
        self.written: Path | None = None
        self.imported_run: str | None = None
        self._import_job: str | None = None

        export = Card("Export a notebook")
        export.add(label(STEPS, "Body", wrap=True))
        row = QHBoxLayout()
        self.source = QComboBox()
        self.source.setPlaceholderText("No experiments yet: run one, or open one in the Experiment Builder")
        row.addWidget(self.source, 1)
        self.export_button = QPushButton(icon("file-plus"), "Write notebook")
        self.export_button.setObjectName("Primary")
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        export.add(row)
        self.export_text = label("", "Mono", wrap=True)
        export.add(self.export_text)
        row = QHBoxLayout()
        self.folder_button = QPushButton(icon("folder-open"), "Open folder")
        self.folder_button.clicked.connect(self._open_folder)
        colab = QPushButton(icon("external-link"), "Open Google Colab")
        colab.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(COLAB_URL)))
        row.addWidget(self.folder_button)
        row.addWidget(colab)
        row.addStretch(1)
        export.add(row)
        self.body.addWidget(export)

        written = Card("Notebooks written", "notebooks/colab/ in this workspace", padded=False)
        self.table = DataTable(("Notebook", "Written"), mono_columns=(1,), stretch_column=0)
        written.add(self.table)
        self.body.addWidget(written)

        back = Card("Import results", "the zip the notebook's last cell downloads")
        row = QHBoxLayout()
        self.import_button = QPushButton(icon("upload"), "Choose zip…")
        self.import_button.clicked.connect(self._choose)
        self.show_button = QPushButton(icon("activity"), "Show on the Training page")
        self.show_button.clicked.connect(self._show_run)
        row.addWidget(self.import_button)
        row.addWidget(self.show_button)
        row.addStretch(1)
        back.add(row)
        self.import_text = label("", "Body", wrap=True)
        back.add(self.import_text)
        self.body.addWidget(back)
        self.body.addStretch(1)

        ctx.colab.imported.connect(self._imported)
        ctx.experiments.run_changed.connect(lambda _: self._fill_sources())
        ctx.experiments.draft_changed.connect(lambda _: self._fill_sources())
        self._fill_sources()
        self._fill_written()
        self._update()

    # Export ---------------------------------------------------------------------------------------
    def _fill_sources(self) -> None:
        keep = self.source.currentData()
        self.source.clear()
        if self.ctx.experiments.draft is not None:
            self.source.addItem(f"Experiment Builder draft · {self.ctx.experiments.draft.name}", "draft")
        for row in self.ctx.store.experiments(limit=100):
            self.source.addItem(f"{row['name']} · {row['task']}", row["id"])
        index = self.source.findData(keep)
        self.source.setCurrentIndex(max(index, 0))
        self._update()

    def spec(self) -> ExperimentSpec | None:
        key = self.source.currentData()
        if key == "draft":
            return self.ctx.experiments.draft
        row = self.ctx.store.experiment(key) if key else None
        try:
            return load_spec_text(row["spec_yaml"]) if row else None
        except SpecError:
            return None

    def export(self) -> Path | None:
        spec = self.spec()
        if spec is None:
            self.export_text.setText("Choose an experiment first (run one, or open one in the builder).")
            return None
        try:
            self.written = self.ctx.colab.export(spec)
        except (KeyError, OSError) as exc:
            self.export_text.setText(f"Could not write the notebook: {exc}")
            return None
        self.export_text.setText(str(self.written))
        self._fill_written()
        self._update()
        return self.written

    def _fill_written(self) -> None:
        self.table.clear_rows()
        for path in self.ctx.colab.notebooks():
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.table.add_row([path.name, stamp])

    def _open_folder(self) -> None:
        folder = self.ctx.colab.folder
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # Import ---------------------------------------------------------------------------------------
    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Colab results", str(Path.home()), "Zip files (*.zip)")
        if path:
            self.start_import(Path(path))

    def start_import(self, archive: Path) -> str | None:
        if self._import_job is not None:
            return None
        self.import_text.setText(f"Importing {archive.name}…")
        self._import_job = self.ctx.colab.start_import(archive)
        self._update()
        return self._import_job

    def _imported(self, job_id: str, run_id: str, error: str) -> None:
        if job_id != self._import_job:
            return
        self._import_job = None
        if error:
            self.import_text.setText(f"Not imported: {error}")
        else:
            self.imported_run = run_id
            view = self.ctx.experiments.view(run_id)
            self.import_text.setText(f"Imported as run {run_id[:8]} ({view.name}, {view.status}).")
        self._update()

    def _show_run(self) -> None:
        if self.imported_run:
            self.ctx.experiments.show(self.imported_run, self.ctx.navigate)

    def _update(self) -> None:
        self.export_button.setEnabled(self.source.count() > 0)
        self.export_button.setToolTip("" if self.source.count() else "Run or design an experiment first")
        self.folder_button.setEnabled(self.ctx.colab.folder.is_dir())
        self.import_button.setEnabled(self._import_job is None)
        self.show_button.setEnabled(self.imported_run is not None)
