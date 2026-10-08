"""Computer Vision lab: what vision tasks the lab runs, the datasets and models for them,
and preset experiments to start from."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Modality, Task
from labs.computer_vision.presets import CV_PRESETS, CvPreset

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, Page, Pill, ResponsiveRow, label
from .builder.choosers import DatasetDownload
from .builder.form import task_title

__all__ = ["ComputerVisionPage"]

_VISION_TASKS = (Task.IMAGE_CLASSIFICATION, Task.OBJECT_DETECTION, Task.SEMANTIC_SEGMENTATION,
                 Task.INSTANCE_SEGMENTATION, Task.OCR, Task.POSE_ESTIMATION, Task.TRACKING)
_MODEL_TONES = {"ready": ("Ready", "ok"), "planned": ("Planned", "planned"), "experimental": ("Experimental",
                                                                                              "experimental")}


class ComputerVisionPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Computer Vision", "image classification now; detection and more from v0.5", parent)
        self.ctx = ctx
        explainer = QPushButton(icon("brain-circuit"), "CNN Explainer")
        explainer.clicked.connect(lambda: ctx.navigate("cnn_explainer"))
        self.head.add_action(explainer)

        self.body.addWidget(self._presets_card())
        self.dataset_panels: dict[str, DatasetDownload] = {}
        self.body.addWidget(self._datasets_card())
        row = ResponsiveRow([(self._tasks_card(), 1), (self._models_card(), 2)], breakpoint=900)
        self.body.addWidget(row)
        self.body.addStretch(1)

    # Cards -------------------------------------------------------------------------
    def _presets_card(self) -> Card:
        card = Card("Start from a preset", "opens in the Experiment Builder")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.preset_buttons: dict[str, QPushButton] = {}
        for index, preset in enumerate(CV_PRESETS):
            tile = QWidget()
            box = QVBoxLayout(tile)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            box.addWidget(label(preset.title, "BodyStrong"))
            box.addWidget(label(preset.summary, "Body", wrap=True))
            box.addWidget(label(preset.cost, "CardCaption", wrap=True))
            button = QPushButton("Open in builder")
            button.clicked.connect(lambda _=False, p=preset: self.open_preset(p))
            row = QHBoxLayout()
            row.addWidget(button)
            row.addStretch(1)
            box.addLayout(row)
            box.addStretch(1)
            self.preset_buttons[preset.id] = button
            grid.addWidget(tile, index // 2, index % 2)
        card.add(grid)
        return card

    def _datasets_card(self) -> Card:
        card = Card("Datasets", "downloaded into this workspace's datasets/ folder")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        cards = [c for c in self.ctx.catalog.datasets.all()
                 if c.modality == Modality.IMAGE and Task.IMAGE_CLASSIFICATION in c.tasks and c.adapter
                 and c.maturity != "planned"]
        for index, dataset in enumerate(cards):
            tile = QWidget()
            box = QVBoxLayout(tile)
            box.setContentsMargins(0, 0, 0, 0)
            panel = DatasetDownload(self.ctx)
            head = QHBoxLayout()
            head.addWidget(label(f"{dataset.name}", "BodyStrong"))
            head.addWidget(label(f"{dataset.size} · {dataset.num_classes} classes", "CardCaption"))
            head.addStretch(1)
            head.addWidget(panel.status)
            box.addLayout(head)
            box.addWidget(label(dataset.description, "Body", wrap=True))
            box.addWidget(panel)
            panel.set_card(dataset.id)
            self.dataset_panels[dataset.id] = panel
            grid.addWidget(tile, index, 0)
        card.add(grid)
        card.add(label("ImageNet, COCO and the other large datasets are listed in the Dataset Hub with "
                       "their terms; the lab never downloads them for you.", "CardCaption", wrap=True))
        return card

    def _tasks_card(self) -> Card:
        card = Card("Tasks", padded=False)
        table = DataTable(("Task", "Status"), stretch_column=0)
        maturity = self.ctx.catalog.runners.task_maturity()
        for task in _VISION_TASKS:
            state = maturity.get(task)
            pill = (Pill("Ready", "ok") if state == "stable" else
                    Pill("Experimental · v0.5", "experimental") if state else Pill("Planned", "planned"))
            table.add_row((task_title(task), pill))
        card.add(table)
        return card

    def _models_card(self) -> Card:
        card = Card("Models", padded=False)
        table = DataTable(("Model", "Tasks", "Licence", "Status"), stretch_column=0)
        for model in self.ctx.catalog.models.all():
            if not set(model.tasks) & set(_VISION_TASKS):
                continue
            status = self.ctx.catalog.models.status(model.id)
            text, tone = _MODEL_TONES.get(status, (status.replace("_", " ").capitalize(), "planned"))
            table.add_row((model.name, ", ".join(task_title(t) for t in model.tasks), model.license.name,
                           Pill(text if status != "planned" else f"Planned · {model.planned_for}", tone)))
        card.add(table)
        return card

    def open_preset(self, preset: CvPreset) -> None:
        self.ctx.experiments.open_in_builder(preset.make(), self.ctx.navigate)
