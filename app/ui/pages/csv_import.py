"""Import a CSV file as a dataset: preview its columns, name it, pick the target column and
the task, then copy it into the workspace (in the background) with a card of its own."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from labs.classical_ml.datasets import CsvPreview, preview_csv

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, Pill, label

__all__ = ["CsvImport"]

#: Rows the preview reads, so choosing a big file stays quick (the import reads all of it).
PREVIEW_ROWS = 100_000
_TASKS = (("Classification (predict a class)", "tabular_classification"),
          ("Regression (predict a number)", "tabular_regression"))


class CsvImport(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Import a CSV file", "copied into this workspace's datasets/imported/ folder",
                         parent=parent)
        self.ctx = ctx
        self.preview: CsvPreview | None = None
        self._job: str | None = None
        row = QHBoxLayout()
        choose = QPushButton(icon("folder-open"), "Choose CSV file…")
        choose.clicked.connect(self._ask)
        self.file_text = label("No file chosen. The first row must hold the column names.", "CardCaption",
                               wrap=True)
        row.addWidget(choose)
        row.addWidget(self.file_text, 1)
        self.add(row)
        self.columns = DataTable(("Column", "Kind", "Distinct values"), mono_columns=(0, 2), stretch_column=0)
        self.add(self.columns)
        self.details = QWidget()  # shown once a file is chosen
        rows = QFormLayout(self.details)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(14)
        self.name = QLineEdit()
        self.name.setMaximumWidth(320)
        self.target = QComboBox()
        self.target.setMaximumWidth(320)
        self.target.currentIndexChanged.connect(lambda _: self._target_changed())
        self.task = QComboBox()
        self.task.setMaximumWidth(320)
        for title, task in _TASKS:
            self.task.addItem(title, task)
        rows.addRow("Name", self.name)
        rows.addRow("Target column", self.target)
        rows.addRow("Task", self.task)
        self.add(self.details)
        actions = QHBoxLayout()
        self.import_button = QPushButton("Import")
        self.import_button.setObjectName("Primary")
        self.import_button.clicked.connect(self.start_import)
        self.state = Pill("", "planned")
        self.state.hide()
        self.message = label("", "Mono", wrap=True)
        actions.addWidget(self.import_button)
        actions.addWidget(self.state)
        actions.addWidget(self.message, 1)
        self.add(actions)
        self.add(label("Your own data is imported with the licence category 'unspecified': the lab knows "
                       "nothing about its terms. Without a target column the table is used for clustering "
                       "and dimensionality reduction.", "CardCaption", wrap=True))
        ctx.downloads.imported.connect(self._imported)
        self._set_enabled(False)

    def _set_enabled(self, on: bool) -> None:
        for widget in (self.columns, self.name, self.target, self.task, self.import_button):
            widget.setEnabled(on and self._job is None)
        for widget in (self.columns, self.details, self.import_button):
            widget.setVisible(on)
        self.message.setVisible(bool(self.message.text()))

    def _ask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a CSV file", str(Path.home()),
                                              "CSV files (*.csv *.txt)")
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> bool:
        """Preview ``path``; ``False`` (with the reason shown) when pandas cannot read it."""
        try:
            preview = preview_csv(path, limit=PREVIEW_ROWS)
        except Exception as exc:  # pandas raises many kinds of parser errors
            self.preview = None
            self.file_text.setText(f"{path.name} could not be read: {exc}")
            self._set_enabled(False)
            return False
        self.preview = preview
        rows = f"{preview.rows:,}" if preview.complete else f"more than {preview.rows:,}"
        self.file_text.setText(f"{path.name}: {rows} rows × {len(preview.columns)} columns")
        self.columns.clear_rows()
        for column in preview.columns:
            self.columns.add_row((column, preview.kinds[column], f"{preview.distinct[column]:,}"))
        self.name.setText(path.stem.replace("_", " ").replace("-", " ").strip().capitalize() or "Dataset")
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("None (clustering and projections only)", None)
        for column in preview.columns:
            self.target.addItem(column, column)
        self.target.setCurrentIndex(self.target.count() - 1)  # the last column is the usual convention
        self.target.blockSignals(False)
        self._target_changed()
        self.state.hide()
        self.message.setText("")
        self._set_enabled(True)
        return True

    def _target_changed(self) -> None:
        target = self.target.currentData()
        self.task.setEnabled(target is not None and self._job is None)
        if target is not None and self.preview is not None:
            self.task.setCurrentIndex(max(self.task.findData(self.preview.guess_task(target)), 0))

    def start_import(self) -> str | None:
        if self.preview is None or self._job is not None:
            return None
        name = self.name.text().strip() or self.preview.path.stem
        self._job = self.ctx.downloads.import_csv(self.preview.path, name=name,
                                                  target=self.target.currentData(),
                                                  task=self.task.currentData())
        self.state.setText("Importing…")
        self.state.set_tone("info")
        self.state.show()
        self._set_enabled(True)
        return self._job

    def _imported(self, job_id: str, card_id: str, error: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        self._set_enabled(self.preview is not None)
        if card_id:
            self.state.setText("Imported")
            self.state.set_tone("ok")
            self.message.setText(f"Dataset {card_id} is ready in the Experiment Builder.")
            self.message.show()
        else:
            self.state.setText("Failed")
            self.state.set_tone("fail")
            self.message.setText(error)
            self.message.show()
