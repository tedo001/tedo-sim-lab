"""The Experiment Builder's settings for quantum classifiers (simulated circuits, trained by epochs)
and for TF-IDF text classifiers (one fit). Model settings — qubits, layers, encoding; n-grams,
vocabulary size — go in the Model options box, which shows what each model takes."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.experiment_engine.spec import ExperimentSpec, QuantumTrainingSpec, SklearnTrainingSpec

from ...widgets import Card, label
from .fields import double, form, spin
from .quantum_device import DeviceCard
from .torch_section import form_widget

__all__ = ["QuantumSection", "TextSection"]


def _stack(*cards: Card) -> QWidget:
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(12)
    for card in cards:
        column.addWidget(card)
    return holder


class QuantumSection:
    """``data`` (rows in the data step) and ``cards`` (training)."""

    OPTIONS = "qubits: 4        # 2 … 10\nlayers: 2        # vqc only\nencoding: angle  # angle or zz"

    def __init__(self) -> None:
        self.data = QWidget()
        rows = form_widget(self.data)
        self.test_fraction = double(0.05, 0.5, 0.25, 2, 0.05)
        self.subset = spin(0, 100_000, 0, special="all")
        rows.addRow("Test share", self.test_fraction)
        rows.addRow("Rows", self.subset)
        rows.addRow("",
                    label("Each feature becomes one qubit's rotation, scaled to 0 … π; with more columns "
                          "than "
                              "qubits the first principal components are used.", "CardCaption", wrap=True))
        card = Card("4 · Training", "the circuit is simulated on the CPU")
        rows = form(card)
        self.epochs = spin(1, 1000, 30)
        self.batch = spin(1, 4096, 16)
        self.lr = double(1e-4, 5.0, 0.1, 4, 0.01)
        self.seed = spin(0, 2_147_483_647, 42)
        for title, widget in (("Epochs", self.epochs), ("Batch size", self.batch), ("Learning rate", self.lr),
                              ("Seed", self.seed)):
            rows.addRow(title, widget)
        card.add(label("The variational classifier learns by parameter-shift gradients and Adam; the kernel "
                       "SVM fits once (epochs are ignored).", "CardCaption", wrap=True))
        self.device = DeviceCard()
        self.cards = _stack(card, self.device)

    def data_fields(self) -> dict[str, Any]:
        return {"subset": self.subset.value() or None}

    def training(self) -> QuantumTrainingSpec:
        return QuantumTrainingSpec(epochs=self.epochs.value(), batch_size=self.batch.value(),
                                   lr=self.lr.value(), test_fraction=self.test_fraction.value(),
                                   simulator=self.device.simulator(),
                                   baselines=self.device.chosen_baselines())

    def load(self, spec: ExperimentSpec) -> None:
        self.subset.setValue(spec.data.subset or 0)
        self.seed.setValue(spec.seed)
        if isinstance(spec.training, QuantumTrainingSpec):
            self.epochs.setValue(spec.training.epochs)
            self.batch.setValue(spec.training.batch_size)
            self.lr.setValue(spec.training.lr)
            self.test_fraction.setValue(spec.training.test_fraction)
            self.device.load(spec.training)


class TextSection:
    """TF-IDF text classification: ``data`` rows and the fitting card."""

    OPTIONS = "max_features: 50000\nngram_max: 2     # 1 = words, 2 = words and pairs\nmin_df: 2"

    def __init__(self) -> None:
        self.data = QWidget()
        rows = form_widget(self.data)
        self.test_fraction = double(0.05, 0.5, 0.2, 2, 0.05)
        self.subset = spin(0, 10_000_000, 0, special="all")
        self.note = label("", "CardCaption", wrap=True)
        rows.addRow("Test share", self.test_fraction)
        rows.addRow("Texts per split", self.subset)
        rows.addRow("", self.note)
        card = Card("4 · Fitting", "one fit on the training texts, scored once on the test texts")
        rows = form(card)
        self.seed = spin(0, 2_147_483_647, 42)
        rows.addRow("Seed", self.seed)
        card.add(label("Words and word pairs are weighted by TF-IDF (rare, telling words count most); the "
                       "results "
                       "list the words that push a text towards each class.", "CardCaption", wrap=True))
        self.cards = _stack(card)

    def set_own_split(self, own: bool) -> None:
        """Datasets with their own test split (20 Newsgroups) ignore the test share."""
        self.test_fraction.setEnabled(not own)
        self.note.setText("This dataset has its own train/test split; it is used as published." if own
                          else "")
        self.note.setVisible(own)

    def data_fields(self) -> dict[str, Any]:
        return {"subset": self.subset.value() or None}

    def training(self) -> SklearnTrainingSpec:
        return SklearnTrainingSpec(test_fraction=self.test_fraction.value())

    def load(self, spec: ExperimentSpec) -> None:
        self.subset.setValue(spec.data.subset or 0)
        self.seed.setValue(spec.seed)
        if isinstance(spec.training, SklearnTrainingSpec):
            self.test_fraction.setValue(spec.training.test_fraction)
