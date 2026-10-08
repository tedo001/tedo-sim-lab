"""Model Registry: models trained in this lab, as numbered versions per name, each with its
metrics, checkpoint, source run, stage and notes (and its MLflow model version when the run
was tracked there). Register a version from a finished run on the Training page."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QMessageBox, QPlainTextEdit, QPushButton, QWidget

from ...services.context import AppContext
from ...services.model_registry import STAGES, RegisteredModel
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, Pill, ResponsiveRow, label
from .home_cards import local_time
from .sklearn_results import METRIC_TITLES
from .training import tabular_score

__all__ = ["ModelRegistryPage"]

STAGE_TONES = {"none": "planned", "staging": "info", "production": "ok", "archived": "planned"}


def stage_pill(stage: str) -> Pill:
    return Pill("No stage" if stage == "none" else stage.capitalize(), STAGE_TONES.get(stage, "planned"))


class ModelRegistryPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Model Registry", "versions, metrics, checkpoint and source run of every model",
                         parent)
        self.ctx = ctx
        self.ids: list[str] = []
        self.selected: str | None = None
        training = QPushButton(icon("activity"), "Training")
        training.clicked.connect(lambda: ctx.navigate("training"))
        self.head.add_action(training)

        listing = Card("Registered models", padded=False)
        self.table = DataTable(("Model", "Version", "Stage", "Score"), mono_columns=(1, 3), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        listing.add(self.table)
        self.empty = label("No models yet. Finish a run, then press Register model on the Training page; "
                           "each registration of the same name adds a version.", "CardCaption", wrap=True)
        self.empty.setContentsMargins(14, 10, 14, 12)
        listing.add(self.empty)

        self.detail = Card("")
        self.stage_holder = QHBoxLayout()
        holder = QWidget()
        holder.setLayout(self.stage_holder)
        self.stage_holder.setContentsMargins(0, 0, 0, 0)
        self.detail.add_head_widget(holder)
        self.values = KeyValues((("Model card", ""), ("Checkpoint", ""), ("Source run", ""), ("MLflow", ""),
                                 ("Registered", "")))
        self.detail.add(self.values)
        self.metrics = DataTable(("Measure", "Value"), mono_columns=(1,), stretch_column=0)
        self.detail.add(self.metrics)
        row = QHBoxLayout()
        self.stage = QComboBox()
        for stage in STAGES:
            self.stage.addItem("No stage" if stage == "none" else stage.capitalize(), stage)
        self.stage.currentIndexChanged.connect(self._stage_changed)
        row.addWidget(label("Stage", "Body"))
        row.addWidget(self.stage)
        row.addWidget(label("Production is one version per name; promoting one archives the other.",
                            "CardCaption", wrap=True), 1)
        self.detail.add(row)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Notes: what this version is for, how it was checked…")
        self.notes.setFixedHeight(80)
        self.detail.add(self.notes)
        buttons = QHBoxLayout()
        save = QPushButton("Save notes")
        save.clicked.connect(self._save_notes)
        self.run_button = QPushButton(icon("activity"), "Open source run")
        self.run_button.clicked.connect(self._open_run)
        self.folder_button = QPushButton(icon("folder-open"), "Open checkpoint folder")
        self.folder_button.clicked.connect(self._open_folder)
        self.delete_button = QPushButton(icon("x"), "Remove version")
        self.delete_button.setToolTip("Forget this version; the checkpoint stays in its run folder")
        self.delete_button.clicked.connect(lambda: self.delete())
        for button in (save, self.run_button, self.folder_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.detail.add(buttons)
        self.body.addWidget(ResponsiveRow([(listing, 2), (self.detail, 3)], breakpoint=1000))
        self.body.addStretch(1)
        ctx.models.registered.connect(self._registered)
        ctx.models.changed.connect(self.refresh)
        self.refresh()

    # Listing --------------------------------------------------------------------------
    def refresh(self) -> None:
        models = self.ctx.models.all()
        self.ids = [model.id for model in models]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for model in models:
            self.table.add_row((model.name, f"v{model.version}", stage_pill(model.stage),
                                tabular_score(model.metrics)))
        self.table.blockSignals(False)
        self.table.setVisible(bool(models))
        self.empty.setVisible(not models)
        if self.selected not in self.ids:
            self.selected = self.ids[0] if self.ids else None
        if self.selected is not None:
            self.table.blockSignals(True)
            self.table.selectRow(self.ids.index(self.selected))
            self.table.blockSignals(False)
        self._show()

    def _registered(self, model_id: str) -> None:
        self.selected = model_id  # show a new version as soon as it exists (refresh follows)

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.ids[rows[0].row()]
            self._show()

    def _model(self) -> RegisteredModel | None:
        return self.ctx.models.get(self.selected) if self.selected else None

    def _show(self) -> None:
        model = self._model()
        self.detail.setVisible(model is not None)
        if model is None:
            return
        self.detail.set_title(model.label)
        while self.stage_holder.count():
            widget = self.stage_holder.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.stage_holder.addWidget(stage_pill(model.stage))
        self.values.set_value("Model card", model.card_id or "—")
        self.values.set_value("Checkpoint", str(model.checkpoint) if model.checkpoint else "—")
        self.values.set_value("Source run", model.source_run_id or "deleted")
        self.values.set_value("MLflow", f"model {model.name!r} version {model.mlflow_version}"
                              if model.mlflow_version else "not registered (run not tracked in MLflow)")
        self.values.set_value("Registered", local_time(model.created_at))
        self.metrics.clear_rows()
        for key, value in model.metrics.items():
            self.metrics.add_row((METRIC_TITLES.get(key, key), f"{value:.4f}"))
        self.metrics.setVisible(bool(model.metrics))
        self.stage.blockSignals(True)
        self.stage.setCurrentIndex(max(self.stage.findData(model.stage), 0))
        self.stage.blockSignals(False)
        self.notes.setPlainText(model.notes)
        self.run_button.setEnabled(model.source_run_id is not None)
        self.folder_button.setEnabled(model.checkpoint is not None and model.checkpoint.parent.is_dir())

    # Actions --------------------------------------------------------------------------
    def _stage_changed(self) -> None:
        if self.selected is not None:
            self.ctx.models.set_stage(self.selected, self.stage.currentData())

    def _save_notes(self) -> None:
        if self.selected is not None:
            self.ctx.models.set_notes(self.selected, self.notes.toPlainText().strip())

    def _open_run(self) -> None:
        model = self._model()
        if model is not None and model.source_run_id:
            self.ctx.experiments.show(model.source_run_id, self.ctx.navigate)

    def _open_folder(self) -> None:
        model = self._model()
        if model is not None and model.checkpoint is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(model.checkpoint.parent)))

    def delete(self, *, confirm: bool = True) -> bool:
        model = self._model()
        if model is None:
            return False
        question = f"Remove {model.label} from the registry? Its checkpoint stays in the run folder."
        answer = QMessageBox.question(self, "Remove version", question) if confirm else None
        if confirm and answer != QMessageBox.StandardButton.Yes:
            return False
        self.ctx.models.delete(model.id)
        return True
