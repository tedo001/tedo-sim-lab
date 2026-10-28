"""The Experiment Builder's numbered steps, turned into an :class:`ExperimentSpec` and back.

Every control maps to one field of ``experiment.yaml``; nothing here runs anything. Image and audio
tasks, and fine-tuned transformers for text, show the PyTorch settings (:mod:`.torch_section`);
tabular tasks the scikit-learn ones (:mod:`.sklearn_section`); TF-IDF text classifiers and quantum
classifiers their own (:mod:`.other_sections`).
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
from .other_sections import QuantumSection, TextSection
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
        task_card.add(label("Image and audio classification train PyTorch networks (spoken digits as "
                            "spectrograms); text classification fits TF-IDF models or fine-tunes a "
                            "downloaded transformer; tabular tasks fit scikit-learn and XGBoost models; "
                            "quantum classification trains simulated circuits. Detection, segmentation, OCR, "
                            "pose and tracking are listed as Experimental (planned for v0.5); pretrained "
                            "detectors run on Model Testing.",
                            "CardCaption", wrap=True))

        self.torch = TorchSection(ctx)
        self.sklearn = SklearnSection(ctx)
        self.quantum = QuantumSection()
        self.text = TextSection()
        data_card = Card("2 · Data")
        self.dataset = DatasetChooser(ctx)
        data_card.add(self.dataset)
        for section in (self.torch, self.sklearn, self.quantum, self.text):
            data_card.add(section.data)

        model_card = Card("3 · Model")
        self.model = ModelChooser(ctx)
        model_card.add(self.model)
        rows = form(model_card)
        self.params = yaml_box("one option per line, e.g.\nn_estimators: 300")
        rows.addRow("Model options", self.params)

        for card in (about, task_card, data_card, model_card):
            layout.addWidget(card)
        for section in (self.torch, self.sklearn, self.quantum, self.text):
            layout.addWidget(section.cards)
        self.task.currentIndexChanged.connect(self._task_changed)
        self.model.changed.connect(self._show_sections)
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

    @property
    def kind(self) -> str:
        """Which settings apply: "torch", "sklearn" (tables), "text" (TF-IDF) or "quantum"."""
        task = self.current_task
        if self.tabular:
            return "sklearn"
        if task == Task.QUANTUM_CLASSIFICATION:
            return "quantum"
        if task == Task.TEXT_CLASSIFICATION:
            card_id = self.model.card_id
            framework = self.ctx.catalog.models.get(card_id).framework.value if card_id else "sklearn"
            return "torch" if framework == "huggingface" else "text"
        return "torch"

    def _task_changed(self) -> None:
        task = self.current_task
        self.dataset.set_task(task)
        self.model.set_task(task)
        self.sklearn.set_task(task)
        self._show_sections()
        self._dataset_changed()

    def _show_sections(self) -> None:
        kind = self.kind
        sections = {"torch": self.torch, "sklearn": self.sklearn, "quantum": self.quantum, "text": self.text}
        for name, section in sections.items():
            section.data.setVisible(name == kind)
            section.cards.setVisible(name == kind)
        self.torch.set_images(self.current_task in (Task.IMAGE_CLASSIFICATION, Task.AUDIO_CLASSIFICATION))
        hints = {"quantum": QuantumSection.OPTIONS, "text": TextSection.OPTIONS}
        self.params.setPlaceholderText(hints.get(kind, "max_length: 256" if kind == "torch" and
                                                 self.current_task == Task.TEXT_CLASSIFICATION
                                                 else "one option per line, e.g.\nn_estimators: 300"))

    def _dataset_changed(self) -> None:
        if self.tabular and self.dataset.card_id != getattr(self, "_columns_of", None):
            self._columns_of = self.dataset.card_id
            self.sklearn.set_dataset(self.dataset.card_id)
        card_id = self.dataset.card_id
        if self.current_task == Task.TEXT_CLASSIFICATION and card_id:
            self.text.set_own_split("test" in self.ctx.catalog.datasets.get(card_id).splits)

    # Spec ⇄ controls ---------------------------------------------------------------
    def spec(self) -> ExperimentSpec:
        """The spec the controls describe; raises ``ValueError`` (pydantic's, or a message about
        the YAML boxes) if it is invalid."""
        task = self.current_task
        params = yaml_mapping(self.params.toPlainText(), "Model options")
        kind = self.kind
        if kind == "sklearn":
            data, training = self.sklearn.data_fields(), self.sklearn.training(task)
            runtime, seed = RuntimeSpec(device="cpu"), self.sklearn.seed.value()
        elif kind in ("quantum", "text"):
            section = self.quantum if kind == "quantum" else self.text
            data, training = section.data_fields(), section.training()
            runtime, seed = RuntimeSpec(device="cpu"), section.seed.value()
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
        self._show_sections()
        self.params.setPlainText(yaml_text(dict(spec.model.params)))
        sections = {"sklearn": self.sklearn, "quantum": self.quantum, "text": self.text, "torch": self.torch}
        sections[self.kind].load(spec)
