"""Deep Learning lab: design an image classifier layer by layer. The page checks the stack
against the chosen dataset's image size as you edit (output shape and parameters after every
layer), draws it, writes it out as PyTorch code, and hands it to the Experiment Builder or
the queue as model ``layer_stack`` with the layers in ``model.params.layers``."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QWidget,
)

from core.common.vocab import Modality, Task
from labs.computer_vision.presets import image_spec
from labs.deep_learning.layers import LayerError, Step, plan, pytorch_code

from ...services.context import AppContext
from ..icons import icon
from ..theme import mono_font
from ..widgets import Card, DataTable, KeyValues, Page, Pill, ResponsiveRow, label
from .deep_learning_parts import LayerEditor, StackDiagram

__all__ = ["DeepLearningPage"]

#: Kinds listed as experimental in this release: shown, not offered.
EXPERIMENTAL = ("RNN", "LSTM", "GRU", "Autoencoder", "Transformer")


class DeepLearningPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Deep Learning", "design a network layer by layer, then train it like any other "
                         "model", parent)
        self.ctx = ctx
        self.steps: list[Step] = []
        builder = QPushButton(icon("sliders-horizontal"), "Experiment Builder")
        builder.clicked.connect(lambda: ctx.navigate("experiment_builder"))
        self.head.add_action(builder)

        data = Card("Input", "the dataset sets the image size and the number of classes")
        row = QHBoxLayout()
        self.dataset = QComboBox()
        for card in ctx.catalog.datasets.all():
            if (card.modality == Modality.IMAGE and Task.IMAGE_CLASSIFICATION in card.tasks and card.adapter
                    and card.maturity != "planned"):
                self.dataset.addItem(f"{card.name} · {card.num_classes} classes", card.id)
        self.dataset.currentIndexChanged.connect(lambda _: self.refresh())
        row.addWidget(self.dataset, 1)
        self.shape_text = label("", "Mono")
        row.addWidget(self.shape_text)
        data.add(row)
        self.data_status = label("", "CardCaption", wrap=True)
        data.add(self.data_status)
        self.body.addWidget(data)

        layers = Card("Layers", "the final Linear layer to the classes is added for you")
        self.editor = LayerEditor()
        self.editor.changed.connect(self.refresh)
        layers.add(self.editor)
        soon = QHBoxLayout()
        soon.addWidget(label("Not in this release:", "CardCaption"))
        for name in EXPERIMENTAL:
            pill = Pill(name, "experimental")
            pill.setToolTip("Sequence and generative layers are planned; a stack here is a feed-forward "
                            "image classifier.")
            soon.addWidget(pill)
        soon.addStretch(1)
        layers.add(soon)

        shapes = Card("Shapes", padded=False)
        self.table = DataTable(("#", "Layer", "Output", "Parameters"), mono_columns=(0, 2, 3),
                               stretch_column=1)
        shapes.add(self.table)
        self.body.addWidget(ResponsiveRow([(layers, 1), (shapes, 1)]))

        summary = Card("Network")
        self.problem = label("", "Problem", wrap=True)
        summary.add(self.problem)
        self.facts = KeyValues([("Layers", "—"), ("Parameters", "—"), ("Weights size", "—"),
                                ("Output", "—")])
        summary.add(self.facts)
        self.diagram = StackDiagram()
        scroll = QScrollArea()
        scroll.setWidget(self.diagram)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMinimumHeight(196)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        summary.add(scroll)
        self.body.addWidget(summary)

        code = Card("PyTorch", "what the lab builds, as a module you can copy")
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self.copy_code)
        code.add_head_widget(self.copy_button)
        self.code = QPlainTextEdit()
        self.code.setReadOnly(True)
        self.code.setFont(mono_font())
        self.code.setMinimumHeight(260)
        code.add(self.code)
        self.body.addWidget(code)

        train = Card("Train it", "as model layer_stack; the layers are saved in experiment.yaml")
        row = QHBoxLayout()
        row.addWidget(label("Name", "Body"))
        self.name = QLineEdit("my-network")
        row.addWidget(self.name, 1)
        row.addWidget(label("Epochs", "Body"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(5)
        row.addWidget(self.epochs)
        self.open_button = QPushButton(icon("sliders-horizontal"), "Open in builder")
        self.open_button.clicked.connect(self.open_in_builder)
        row.addWidget(self.open_button)
        self.run_button = QPushButton(icon("play"), "Train now")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self.train)
        row.addWidget(self.run_button)
        train.add(row)
        self.train_text = label("", "CardCaption", wrap=True)
        train.add(self.train_text)
        self.body.addWidget(train)
        self.body.addStretch(1)
        ctx.downloads.finished.connect(lambda *_: self.refresh())
        self.refresh()

    # State ------------------------------------------------------------------------------------
    def input_shape(self) -> tuple[int, int, int, int]:
        """(channels, height, width, classes) of the chosen dataset."""
        card_id = self.dataset.currentData()
        card = self.ctx.catalog.datasets.get(card_id)
        adapter = self.ctx.catalog.datasets.adapter(card_id)
        height, width = adapter.native_size
        return adapter.in_channels, height, width, card.num_classes or 0

    def refresh(self) -> None:
        if self.dataset.currentData() is None:
            return
        channels, height, width, classes = self.input_shape()
        self.shape_text.setText(f"{channels}×{height}×{width} → {classes}")
        ready = self.ctx.catalog.datasets.status(self.dataset.currentData()) == "ready"
        self.data_status.setText("Downloaded." if ready else
                                 "Not downloaded yet: the Computer Vision lab or the Dataset Hub downloads "
                                 "it. You can still design the network and open it in the builder.")
        self.table.clear_rows()
        try:
            self.steps = plan(self.editor.layers, in_channels=channels, input_size=(height, width),
                              num_classes=classes)
        except LayerError as error:
            self.steps = []
            self.problem.setText(str(error))
        else:
            self.problem.setText("")
        self.problem.setVisible(bool(self.problem.text()))
        for step in self.steps:
            title = step.describe() + ("  (added)" if step.automatic else "")
            self.table.add_row([str(step.index + 1), title, " × ".join(map(str, step.shape)),
                                f"{step.parameters:,}"])
        total = sum(step.parameters for step in self.steps)
        self.facts.set_value("Layers", str(len(self.steps)) if self.steps else "—")
        self.facts.set_value("Parameters", f"{total:,}" if self.steps else "—")
        self.facts.set_value("Weights size", f"{total * 4 / 1e6:.2f} MB (float32)" if self.steps else "—")
        self.facts.set_value("Output", f"{classes} class scores" if self.steps else "—")
        self.diagram.set_steps(self.steps, (channels, height, width))
        self.code.setPlainText(pytorch_code(self.steps, in_channels=channels) if self.steps else
                               "# Fix the stack above to see its PyTorch code.")
        self.copy_button.setEnabled(bool(self.steps))
        self.open_button.setEnabled(bool(self.steps))
        self.run_button.setEnabled(bool(self.steps) and ready)
        self.run_button.setToolTip("" if ready else "Download the dataset first")

    # Actions ----------------------------------------------------------------------------------
    def spec(self):
        name = self.name.text().strip() or "my-network"
        return image_spec(name, self.dataset.currentData(), "layer_stack", self.epochs.value(),
                          params={"layers": [dict(layer) for layer in self.editor.layers]})

    def open_in_builder(self) -> None:
        if self.steps:
            self.ctx.experiments.open_in_builder(self.spec(), self.ctx.navigate)

    def train(self) -> str | None:
        if not self.run_button.isEnabled():
            return None
        try:
            run_id = self.ctx.experiments.launch(self.spec())
        except (ValueError, OSError) as error:
            self.train_text.setText(f"Could not queue it: {error}")
            return None
        self.train_text.setText(f"Queued as run {run_id[:8]}; follow it on the Training page.")
        return run_id

    def copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.code.toPlainText())
