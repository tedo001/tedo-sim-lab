"""The Experiment Builder's numbered steps, turned into an :class:`ExperimentSpec` and back.

Every control maps to one field of ``experiment.yaml``; nothing here runs anything. Image
tasks show the PyTorch settings (:mod:`.torch_section`), tabular tasks the scikit-learn ones
(:mod:`.sklearn_section`).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.common.vocab import TABULAR_TASKS, Task
from core.experiment_engine.spec import DataSpec, ExperimentSpec, ModelSpec, RuntimeSpec

from ....services.context import AppContext
from ...widgets import Card, label
from .choosers import DatasetChooser, ModelChooser
from .fields import form, yaml_box, yaml_mapping, yaml_text
from .sklearn_section import SklearnSection
from .torch_section import MONITORS, TorchSection

__all__ = ["MONITORS", "SpecForm", "task_title"]


def task_title(task: Task) -> str:
    return task.value.replace("_", " ").capitalize()


class SpecForm(QWidget):
    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        about = Card("Experiment")
        rows = form(about)
        self.name = QLineEdit("mnist-simple-cnn")
        self.description = QLineEdit()
        self.description.setPlaceholderText("What you want to find out (optional)")
        rows.addRow("Name", self.name)
        rows.addRow("Description", self.description)

        task_card = Card("1 · Task")
        maturity = ctx.catalog.runners.task_maturity()
        self.task = QComboBox()
        self.task.setMaximumWidth(320)
        for task in Task:
            state = maturity.get(task)
            if state is None:
                continue
            self.task.addItem(task_title(task) + ("" if state == "stable" else f" · {state}"), task)
            if state != "stable":
                self.task.model().item(self.task.count() - 1).setEnabled(False)
        task_card.add(self.task)
        task_card.add(label("Image classification trains PyTorch networks; tabular classification, "
                            "regression, clustering and dimensionality reduction fit scikit-learn and "
                            "XGBoost models. Detection, segmentation, OCR, pose and tracking are listed as "
                            "Experimental (planned for v0.5).", "CardCaption", wrap=True))

        self.torch = TorchSection(ctx)
        self.sklearn = SklearnSection(ctx)
        data_card = Card("2 · Data")
        self.dataset = DatasetChooser(ctx)
        data_card.add(self.dataset)
        data_card.add(self.torch.data)
        data_card.add(self.sklearn.data)

        model_card = Card("3 · Model")
        self.model = ModelChooser(ctx)
        model_card.add(self.model)
        rows = form(model_card)
        self.params = yaml_box("one option per line, e.g.\nn_estimators: 300")
        rows.addRow("Model options", self.params)

        for card in (about, task_card, data_card, model_card):
            layout.addWidget(card)
        layout.addWidget(self.torch.cards)
        layout.addWidget(self.sklearn.cards)
        self.task.currentIndexChanged.connect(self._task_changed)
        self.dataset.changed.connect(self._dataset_changed)
        ctx.downloads.imported.connect(lambda *_: self._task_changed())
        self._connect_changes()
        self._task_changed()
        self.dataset.select("mnist")
        self.model.select("simple_cnn")

    def _connect_changes(self) -> None:
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.changed)
        for widget in self.findChildren(QPlainTextEdit):
            widget.textChanged.connect(self.changed)
        for widget in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self.changed)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.changed)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self.changed)
        for widget in self.findChildren(QListWidget):
            widget.itemChanged.connect(self.changed)
        self.dataset.changed.connect(self.changed)
        self.model.changed.connect(self.changed)

    @property
    def current_task(self) -> Task:
        return self.task.currentData() or Task.IMAGE_CLASSIFICATION

    @property
    def tabular(self) -> bool:
        return self.current_task in TABULAR_TASKS

    def _task_changed(self) -> None:
        task = self.current_task
        self.dataset.set_task(task)
        self.model.set_task(task)
        self.sklearn.set_task(task)
        for widget in (self.torch.data, self.torch.cards):
            widget.setVisible(not self.tabular)
        for widget in (self.sklearn.data, self.sklearn.cards):
            widget.setVisible(self.tabular)
        self._dataset_changed()

    def _dataset_changed(self) -> None:
        if self.tabular and self.dataset.card_id != getattr(self, "_columns_of", None):
            self._columns_of = self.dataset.card_id
            self.sklearn.set_dataset(self.dataset.card_id)

    # Spec ⇄ controls ---------------------------------------------------------------
    def spec(self) -> ExperimentSpec:
        """The spec the controls describe; raises ``ValueError`` (pydantic's, or a message about
        the YAML boxes) if it is invalid."""
        task = self.current_task
        params = yaml_mapping(self.params.toPlainText(), "Model options")
        if self.tabular:
            data, training = self.sklearn.data_fields(), self.sklearn.training(task)
            runtime, seed = RuntimeSpec(device="cpu"), self.sklearn.seed.value()
        else:
            data, training = self.torch.data_fields(), self.torch.training()
            runtime, seed = self.torch.runtime(), self.torch.seed.value()
        return ExperimentSpec(
            name=self.name.text().strip() or "experiment", description=self.description.text().strip(),
            task=task, seed=seed, data=DataSpec(dataset=self.dataset.card_id or "", **data),
            model=ModelSpec(model=self.model.card_id or "", pretrained=self.model.pretrained, params=params),
            training=training, runtime=runtime)

    def load(self, spec: ExperimentSpec) -> None:
        """Fill every control from ``spec`` (fields the form has no control for are dropped)."""
        self.name.setText(spec.name)
        self.description.setText(spec.description)
        self.task.setCurrentIndex(max(self.task.findData(spec.task), 0))
        self._task_changed()
        self.dataset.select(spec.data.dataset)
        self._dataset_changed()
        self.model.select(spec.model.model, spec.model.pretrained)
        self.params.setPlainText(yaml_text(dict(spec.model.params)))
        if spec.task in TABULAR_TASKS:
            self.sklearn.load(spec)
        else:
            self.torch.load(spec)
