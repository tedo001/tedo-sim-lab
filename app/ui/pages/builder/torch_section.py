"""The Experiment Builder's PyTorch settings: image data options, training and device."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from core.experiment_engine.spec import (
    CheckpointSpec,
    EarlyStoppingSpec,
    ExperimentSpec,
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
from .fields import combo, double, form, spin

__all__ = ["MONITORS", "TorchSection"]

#: Checkpoint choices: (title, metric, mode).
MONITORS = (("Lowest validation loss", "val_loss", "min"), ("Highest validation accuracy", "val_acc", "max"),
            ("Highest validation F1", "val_f1", "max"))


class TorchSection:
    """Builds ``data`` (rows inside the data step) and ``cards`` (training and device steps)."""

    def __init__(self, ctx: AppContext) -> None:
        self.data = QWidget()
        rows = form_widget(self.data)
        self.val_fraction = double(0.0, 0.5, 0.1, 2, 0.05)
        self.subset = spin(0, 1_000_000, 0, special="all")
        self.normalize = QCheckBox("Normalise with the dataset's mean and standard deviation")
        self.normalize.setChecked(True)
        rows.addRow("Validation share", self.val_fraction)
        rows.addRow("Images per split", self.subset)
        rows.addRow("", self.normalize)
        self.augment: dict[str, QCheckBox] = {}
        for name, text in AUGMENTATIONS.items():
            box = QCheckBox(text)
            self.augment[name] = box
            rows.addRow("Augment" if len(self.augment) == 1 else "", box)

        training_card = Card("4 · Training")
        rows = form(training_card)
        self.epochs = spin(1, 1000, 5)
        self.batch = spin(1, 4096, 64)
        self.optimizer = combo([("Adam", "adam"), ("AdamW", "adamw"), ("SGD with momentum", "sgd")])
        self.lr = double(1e-6, 10.0, 1e-3, 6, 1e-4)
        self.weight_decay = double(0.0, 1.0, 0.0, 6, 1e-4)
        self.scheduler = combo([("Constant", "none"), ("Step (÷10 every 5 epochs)", "step"),
                                ("Cosine", "cosine"), ("On plateau", "plateau"), ("One-cycle", "onecycle")])
        self.smoothing = double(0.0, 0.5, 0.0, 2, 0.05)
        self.monitor = combo([(title, f"{key}:{mode}") for title, key, mode in MONITORS])
        self.early = QCheckBox("Stop early")
        self.patience = spin(1, 100, 3)
        self.max_steps = spin(0, 1_000_000, 0, special="all")
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
            rows.addRow(title, widget)
        rows.addRow("", early_row)

        runtime_card = Card("5 · Device")
        rows = form(runtime_card)
        self.device = combo([("Automatic (GPU when available)", "auto"), ("CPU", "cpu")])
        info = ctx.hardware.info
        if info is not None and info.cuda_available:
            for gpu in info.gpus:
                self.device.addItem(f"cuda:{gpu.index} · {gpu.name}", f"cuda:{gpu.index}")
        if info is not None and info.mps_available:
            self.device.addItem("Apple MPS", "mps")
        self.precision = combo([("fp32", "fp32"), ("fp16 (CUDA, halves memory)", "fp16"), ("bf16", "bf16")])
        self.seed = spin(0, 2_147_483_647, 42)
        self.deterministic = QCheckBox("Deterministic algorithms (slower, repeatable on the same machine)")
        self.workers = spin(0, 32, 0)
        for title, widget in (("Device", self.device), ("Precision", self.precision), ("Seed", self.seed),
                              ("", self.deterministic), ("Loader workers", self.workers)):
            rows.addRow(title, widget)
        runtime_card.add(label("Defaults fit a 4 GB GPU: batch 64 at fp32 for these models.", "CardCaption",
                               wrap=True))
        self.cards = QWidget()
        stack = QVBoxLayout(self.cards)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(12)
        stack.addWidget(training_card)
        stack.addWidget(runtime_card)

    # Spec ⇄ controls ---------------------------------------------------------------
    def data_fields(self) -> dict[str, Any]:
        return {"split": SplitSpec(val_fraction=self.val_fraction.value()),
                "preprocessing": (TransformSpec(name="normalize"),) if self.normalize.isChecked() else (),
                "augmentation": tuple(TransformSpec(name=name) for name, box in self.augment.items()
                                      if box.isChecked()),
                "subset": self.subset.value() or None}

    def training(self) -> TorchTrainingSpec:
        monitor, mode = self.monitor.currentData().split(":")
        return TorchTrainingSpec(
            epochs=self.epochs.value(), batch_size=self.batch.value(),
            optimizer=OptimizerSpec(name=self.optimizer.currentData(), lr=self.lr.value(),
                                    weight_decay=self.weight_decay.value()),
            scheduler=SchedulerSpec(name=self.scheduler.currentData()),
            label_smoothing=self.smoothing.value(), checkpoint=CheckpointSpec(monitor=monitor, mode=mode),
            early_stopping=EarlyStoppingSpec(monitor=monitor, mode=mode, patience=self.patience.value())
            if self.early.isChecked() else None,
            max_steps_per_epoch=self.max_steps.value() or None)

    def runtime(self) -> RuntimeSpec:
        return RuntimeSpec(device=self.device.currentData(), precision=self.precision.currentData(),
                           num_workers=self.workers.value(), deterministic=self.deterministic.isChecked())

    def load(self, spec: ExperimentSpec) -> None:
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


def form_widget(widget: QWidget) -> Any:
    from PySide6.QtWidgets import QFormLayout

    rows = QFormLayout(widget)
    rows.setContentsMargins(0, 0, 0, 0)
    rows.setHorizontalSpacing(14)
    rows.setVerticalSpacing(8)
    return rows
