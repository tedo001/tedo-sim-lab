"""The Experiment Builder's numbered steps, turned into an :class:`ExperimentSpec` and back.

Every control maps to one field of ``experiment.yaml``; nothing here runs anything.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.common.vocab import TABULAR_TASKS, Task
from core.experiment_engine.spec import (
    CheckpointSpec,
    DataSpec,
    EarlyStoppingSpec,
    ExperimentSpec,
    ModelSpec,
    OptimizerSpec,
    RuntimeSpec,
    SchedulerSpec,
    SplitSpec,
    TorchTrainingSpec,
    TransformSpec,
)
from labs.computer_vision.training.data import AUGMENTATIONS

from ....services.context import AppContext
from ...widgets import Card, label
from .choosers import DatasetChooser, ModelChooser

__all__ = ["MONITORS", "SpecForm", "task_title"]

#: Checkpoint choices: (title, metric, mode).
MONITORS = (("Lowest validation loss", "val_loss", "min"), ("Highest validation accuracy", "val_acc", "max"),
            ("Highest validation F1", "val_f1", "max"))


def task_title(task: Task) -> str:
    return task.value.replace("_", " ").capitalize()


def _spin(low: int, high: int, value: int, *, special: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    if special:
        spin.setSpecialValueText(special)
    spin.setMaximumWidth(160)
    return spin


def _double(low: float, high: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setMaximumWidth(160)
    return spin


def _combo(items: list[tuple[str, object]]) -> QComboBox:
    combo = QComboBox()
    for title, data in items:
        combo.addItem(title, data)
    combo.setMaximumWidth(320)
    return combo


def _form(card: Card) -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(8)
    card.add(form)
    return form


class SpecForm(QWidget):
    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        about = Card("Experiment")
        form = _form(about)
        self.name = QLineEdit("mnist-simple-cnn")
        self.description = QLineEdit()
        self.description.setPlaceholderText("What you want to find out (optional)")
        form.addRow("Name", self.name)
        form.addRow("Description", self.description)

        task_card = Card("1 · Task")
        maturity = ctx.catalog.runners.task_maturity()
        self.task = QComboBox()
        self.task.setMaximumWidth(320)
        notes = []
        for task in Task:
            state = maturity.get(task)
            if task in TABULAR_TASKS:
                state = state or "phase 6"
            if state is None:
                continue
            self.task.addItem(task_title(task) + ("" if state == "stable" else f" · {state}"), task)
            if state != "stable":
                self.task.model().item(self.task.count() - 1).setEnabled(False)
                notes.append(task_title(task))
        task_card.add(self.task)
        task_card.add(label("Only image classification runs in this build. Detection, segmentation, OCR, "
                            "pose and tracking are listed as Experimental (planned for v0.5); tabular "
                            "tasks arrive with the Classical ML lab (build phase 6).", "CardCaption",
                            wrap=True))

        data_card = Card("2 · Data")
        self.dataset = DatasetChooser(ctx)
        data_card.add(self.dataset)
        form = _form(data_card)
        self.val_fraction = _double(0.0, 0.5, 0.1, 2, 0.05)
        self.subset = _spin(0, 1_000_000, 0, special="all")
        self.normalize = QCheckBox("Normalise with the dataset's mean and standard deviation")
        self.normalize.setChecked(True)
        form.addRow("Validation share", self.val_fraction)
        form.addRow("Images per split", self.subset)
        form.addRow("", self.normalize)
        self.augment: dict[str, QCheckBox] = {}
        for name, text in AUGMENTATIONS.items():
            box = QCheckBox(text)
            self.augment[name] = box
            form.addRow("Augment" if len(self.augment) == 1 else "", box)

        model_card = Card("3 · Model")
        self.model = ModelChooser(ctx)
        model_card.add(self.model)

        training_card = Card("4 · Training")
        form = _form(training_card)
        self.epochs = _spin(1, 1000, 5)
        self.batch = _spin(1, 4096, 64)
        self.optimizer = _combo([("Adam", "adam"), ("AdamW", "adamw"), ("SGD with momentum", "sgd")])
        self.lr = _double(1e-6, 10.0, 1e-3, 6, 1e-4)
        self.weight_decay = _double(0.0, 1.0, 0.0, 6, 1e-4)
        self.scheduler = _combo([("Constant", "none"), ("Step (÷10 every 5 epochs)", "step"),
                                 ("Cosine", "cosine"), ("On plateau", "plateau"), ("One-cycle", "onecycle")])
        self.smoothing = _double(0.0, 0.5, 0.0, 2, 0.05)
        self.monitor = _combo([(title, f"{key}:{mode}") for title, key, mode in MONITORS])
        self.early = QCheckBox("Stop early")
        self.patience = _spin(1, 100, 3)
        self.max_steps = _spin(0, 1_000_000, 0, special="all")
        early_row = QHBoxLayout()
        early_row.addWidget(self.early)
        early_row.addWidget(label("after", "Body"))
        early_row.addWidget(self.patience)
        early_row.addWidget(label("epochs without improvement", "Body"))
        early_row.addStretch(1)
        for title, widget in (("Epochs", self.epochs), ("Batch size", self.batch),
                              ("Optimiser", self.optimizer),
                              ("Learning rate", self.lr), ("Weight decay", self.weight_decay),
                              ("Schedule", self.scheduler), ("Label smoothing", self.smoothing),
                              ("Keep the checkpoint with", self.monitor),
                              ("Batches per epoch", self.max_steps)):
            form.addRow(title, widget)
        form.addRow("", early_row)

        runtime_card = Card("5 · Device")
        form = _form(runtime_card)
        self.device = _combo([("Automatic (GPU when available)", "auto"), ("CPU", "cpu")])
        info = ctx.hardware.info
        if info is not None and info.cuda_available:
            for gpu in info.gpus:
                self.device.addItem(f"cuda:{gpu.index} · {gpu.name}", f"cuda:{gpu.index}")
        if info is not None and info.mps_available:
            self.device.addItem("Apple MPS", "mps")
        self.precision = _combo([("fp32", "fp32"), ("fp16 (CUDA, halves memory)", "fp16"), ("bf16", "bf16")])
        self.seed = _spin(0, 2_147_483_647, 42)
        self.deterministic = QCheckBox("Deterministic algorithms (slower, repeatable on the same machine)")
        self.workers = _spin(0, 32, 0)
        for title, widget in (("Device", self.device), ("Precision", self.precision), ("Seed", self.seed),
                              ("", self.deterministic), ("Loader workers", self.workers)):
            form.addRow(title, widget)
        runtime_card.add(label("Defaults fit a 4 GB GPU: batch 64 at fp32 for these models.", "CardCaption",
                               wrap=True))

        for card in (about, task_card, data_card, model_card, training_card, runtime_card):
            layout.addWidget(card)
        self.task.currentIndexChanged.connect(self._task_changed)
        self._connect_changes()
        self._task_changed()
        self.dataset.select("mnist")
        self.model.select("simple_cnn")

    def _connect_changes(self) -> None:
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.changed)
        for widget in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self.changed)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.changed)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self.changed)
        self.dataset.changed.connect(self.changed)
        self.model.changed.connect(self.changed)

    def _task_changed(self) -> None:
        task = self.task.currentData() or Task.IMAGE_CLASSIFICATION
        self.dataset.set_task(task)
        self.model.set_task(task)

    # Spec ⇄ controls ---------------------------------------------------------------
    def spec(self) -> ExperimentSpec:
        """The spec the controls describe; raises ``ValueError`` (pydantic) if it is invalid."""
        monitor, mode = self.monitor.currentData().split(":")
        augmentation = tuple(TransformSpec(name=name) for name, box in self.augment.items()
                             if box.isChecked())
        preprocessing = (TransformSpec(name="normalize"),) if self.normalize.isChecked() else ()
        return ExperimentSpec(
            name=self.name.text().strip() or "experiment", description=self.description.text().strip(),
            task=self.task.currentData() or Task.IMAGE_CLASSIFICATION, seed=self.seed.value(),
            data=DataSpec(dataset=self.dataset.card_id or "", augmentation=augmentation,
                          split=SplitSpec(val_fraction=self.val_fraction.value()),
                          preprocessing=preprocessing,
                          subset=self.subset.value() or None),
            model=ModelSpec(model=self.model.card_id or "", pretrained=self.model.pretrained),
            training=TorchTrainingSpec(
                epochs=self.epochs.value(), batch_size=self.batch.value(),
                optimizer=OptimizerSpec(name=self.optimizer.currentData(), lr=self.lr.value(),
                                        weight_decay=self.weight_decay.value()),
                scheduler=SchedulerSpec(name=self.scheduler.currentData()),
                label_smoothing=self.smoothing.value(), checkpoint=CheckpointSpec(monitor=monitor, mode=mode),
                early_stopping=EarlyStoppingSpec(monitor=monitor, mode=mode, patience=self.patience.value())
                if self.early.isChecked() else None,
                max_steps_per_epoch=self.max_steps.value() or None),
            runtime=RuntimeSpec(device=self.device.currentData(), precision=self.precision.currentData(),
                                num_workers=self.workers.value(),
                                deterministic=self.deterministic.isChecked()))

    def load(self, spec: ExperimentSpec) -> None:
        """Fill every control from ``spec`` (fields the form has no control for are dropped)."""
        self.name.setText(spec.name)
        self.description.setText(spec.description)
        self.task.setCurrentIndex(max(self.task.findData(spec.task), 0))
        self._task_changed()
        self.dataset.select(spec.data.dataset)
        self.model.select(spec.model.model, spec.model.pretrained)
        self.val_fraction.setValue(spec.data.split.val_fraction)
        self.subset.setValue(spec.data.subset or 0)
        self.normalize.setChecked(any(step.name == "normalize" for step in spec.data.preprocessing))
        chosen = {step.name for step in spec.data.augmentation}
        for name, box in self.augment.items():
            box.setChecked(name in chosen)
        training = spec.training
        if isinstance(training, TorchTrainingSpec):
            self.epochs.setValue(training.epochs)
            self.batch.setValue(training.batch_size)
            self.optimizer.setCurrentIndex(max(self.optimizer.findData(training.optimizer.name), 0))
            self.lr.setValue(training.optimizer.lr)
            self.weight_decay.setValue(training.optimizer.weight_decay)
            self.scheduler.setCurrentIndex(max(self.scheduler.findData(training.scheduler.name), 0))
            self.smoothing.setValue(training.label_smoothing)
            watched = f"{training.checkpoint.monitor}:{training.checkpoint.mode}"
            self.monitor.setCurrentIndex(max(self.monitor.findData(watched), 0))
            self.early.setChecked(training.early_stopping is not None)
            if training.early_stopping:
                self.patience.setValue(training.early_stopping.patience)
            self.max_steps.setValue(training.max_steps_per_epoch or 0)
        runtime = spec.runtime
        self.device.setCurrentIndex(max(self.device.findData(runtime.device), 0))
        self.precision.setCurrentIndex(max(self.precision.findData(runtime.precision), 0))
        self.seed.setValue(spec.seed)
        self.deterministic.setChecked(runtime.deterministic)
        self.workers.setValue(runtime.num_workers)
