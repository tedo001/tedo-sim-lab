"""Classical ML lab: preset experiments on scikit-learn's tables, CSV import, and the tabular
datasets and models the lab can use. Runs go through the same queue, tracking and pages as
neural networks."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import TABULAR_TASKS, Modality
from labs.classical_ml.presets import CLASSICAL_PRESETS, ClassicalPreset

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, Page, Pill, label
from .builder.form import task_title
from .csv_import import CsvImport

__all__ = ["ClassicalMlPage"]

#: Short task words for the tables (every task here is tabular).
_SHORT = {"tabular_classification": "classification", "tabular_regression": "regression",
          "clustering": "clustering", "dimensionality_reduction": "projection"}
_DATASET_TONES = {"ready": ("Ready", "ok"), "not_downloaded": ("Missing", "warn"),
                  "not_installed": ("Unavailable", "fail"), "planned": ("Planned", "planned")}


class ClassicalMlPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Classical ML", "scikit-learn and XGBoost on tables, tracked like any run", parent)
        self.ctx = ctx
        builder = QPushButton(icon("sliders-horizontal"), "Experiment Builder")
        builder.clicked.connect(lambda: ctx.navigate("experiment_builder"))
        self.head.add_action(builder)

        self.body.addWidget(self._presets_card())
        self.importer = CsvImport(ctx)
        self.body.addWidget(self.importer)
        datasets = Card("Datasets", "scikit-learn's bundled tables and your imported CSV files", padded=False)
        self.datasets = DataTable(("Dataset", "Size", "Tasks", "Licence", "Status"), mono_columns=(1,),
                                  stretch_column=0)
        datasets.add(self.datasets)
        self.body.addWidget(datasets)
        models = Card("Models", padded=False)
        self.models = DataTable(("Model", "Tasks", "Licence", "Status"), stretch_column=0)
        models.add(self.models)
        self.body.addWidget(models)
        self.body.addStretch(1)
        ctx.downloads.imported.connect(lambda *_: self.refresh())
        self.refresh()

    def _presets_card(self) -> Card:
        card = Card("Start from a preset", "each runs in seconds on a CPU")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.preset_buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        for index, preset in enumerate(CLASSICAL_PRESETS):
            tile = QWidget()
            box = QVBoxLayout(tile)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            box.addWidget(label(preset.title, "BodyStrong"))
            box.addWidget(label(preset.summary, "Body", wrap=True))
            box.addWidget(label(task_title(preset.make().task), "CardCaption"))
            run = QPushButton(icon("play"), "Run")
            run.clicked.connect(lambda _=False, p=preset: self.run_preset(p))
            edit = QPushButton("Open in builder")
            edit.clicked.connect(lambda _=False, p=preset: self.open_preset(p))
            row = QHBoxLayout()
            row.addWidget(run)
            row.addWidget(edit)
            row.addStretch(1)
            box.addLayout(row)
            box.addStretch(1)
            self.preset_buttons[preset.id] = (run, edit)
            grid.addWidget(tile, index // 2, index % 2)
        card.add(grid)
        return card

    def refresh(self) -> None:
        catalog = self.ctx.catalog
        self.datasets.clear_rows()
        for card in catalog.datasets.all():
            if card.modality != Modality.TABULAR or not set(card.tasks) & TABULAR_TASKS:
                continue
            text, tone = _DATASET_TONES.get(catalog.datasets.status(card.id), ("Unknown", "planned"))
            tasks = ", ".join(_SHORT.get(t, t) for t in card.tasks)
            self.datasets.add_row((card.name, card.size or "—", tasks,
                                   card.license.name, Pill(text, tone)))
        self.models.clear_rows()
        for model in catalog.models.all():
            if not set(model.tasks) & TABULAR_TASKS:
                continue
            status = catalog.models.status(model.id)
            pill = Pill("Ready", "ok") if status == "ready" else Pill(status.replace("_", " ").capitalize(),
                                                                       "warn")
            tasks = ", ".join(_SHORT.get(t, t) for t in model.tasks)
            self.models.add_row((model.name, tasks, model.license.name, pill))
        for preset in CLASSICAL_PRESETS:
            run, _edit = self.preset_buttons[preset.id]
            problems = self.ctx.experiments.problems(preset.make())
            run.setEnabled(not problems)
            run.setToolTip("\n".join(problems))

    def open_preset(self, preset: ClassicalPreset) -> None:
        self.ctx.experiments.open_in_builder(preset.make(), self.ctx.navigate)

    def run_preset(self, preset: ClassicalPreset) -> str | None:
        spec = preset.make()
        if self.ctx.experiments.problems(spec):
            return None
        run_id = self.ctx.experiments.launch(spec)
        self.ctx.navigate("training")
        return run_id
