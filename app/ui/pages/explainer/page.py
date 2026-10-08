"""CNN Explainer: see how a convolutional network turns an image into a prediction.

A native port of CNN Explainer (Wang et al., 2020; MIT licence, see
``labs/computer_vision/explainer/LICENSE-cnn-explainer.txt``) for the lab's networks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labs.computer_vision.explainer import (
    PRESETS,
    SCALES,
    ExplainerNet,
    Preset,
    Trace,
    map_limits,
    preset,
    run,
    tiny_vgg,
)

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, Page, Pill, ResponsiveRow, label
from ...widgets.markdown import MarkdownView
from .article import ARTICLE, CREDIT
from .detail_conv import ConvDetail
from .detail_layers import InputDetail, PoolDetail, ReluDetail
from .detail_output import OutputDetail
from .inputs import IMAGE_FILTER, DrawPad, SampleStrip, load_image
from .overview import INPUT, NetworkOverview, describe
from .playground import Playground
from .trained import EXPLAINER_WEIGHTS, preset_for, trained_meta

__all__ = ["ExplainerPage"]

_HOVER_HINT = "Hover a map to see what feeds it; click it to see the arithmetic."


class _Article(MarkdownView):
    """Markdown that grows to its full height, so the page scrolls instead of the text."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.set_markdown(text)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.document().setTextWidth(self.viewport().width())
        self.setFixedHeight(int(self.document().size().height()) + 12)


class ExplainerPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("CNN Explainer", "after CNN Explainer (Wang et al., 2020)", parent)
        self.ctx = ctx
        self.seed = 0
        self.preset: Preset = PRESETS[0]
        self.net: ExplainerNet | None = None
        self.trace: Trace | None = None
        self.image: np.ndarray | None = None
        self.selection: tuple[int, int] = (0, 0)
        self.open_button = QPushButton(icon("image-up"), "Open image…")
        self.open_button.setToolTip("Use your own picture: it is cropped to a square and scaled to the "
                                    "network's input size")
        self.open_button.clicked.connect(self._ask_for_image)
        self.head.add_action(self.open_button)

        self.body.addWidget(self._controls_card())
        self.network_card = Card("Network")
        self.prediction = Pill("", "planned")
        self.network_card.add_head_widget(self.prediction)
        self.overview = NetworkOverview()
        self.overview.node_clicked.connect(self.select)
        self.hover_line = label(_HOVER_HINT, "Mono", wrap=True)
        self.overview.hover_changed.connect(lambda text: self.hover_line.setText(text or _HOVER_HINT))
        self.network_card.add(self.overview)
        self.network_card.add(self.hover_line)
        self.body.addWidget(self.network_card)

        self.detail_card = Card("Detail")
        self.detail_facts = label("", "Mono", wrap=True)
        self.detail_card.add(self.detail_facts)
        self.detail_box = QVBoxLayout()
        self.detail_card.add(self.detail_box)
        self.detail: QWidget | None = None
        self.body.addWidget(self.detail_card)

        playground = Card("Hyperparameters", "how padding, kernel size and stride shape a convolution")
        self.playground = Playground()
        playground.add(self.playground)
        self.body.addWidget(playground)

        article = Card("How a CNN works")
        article.add(_Article(ARTICLE))
        article.add(label(CREDIT, "CardCaption", wrap=True))
        self.body.addWidget(article)
        self.body.addStretch(1)
        self.trained_run: str | None = None
        self.fill_networks()
        experiments = ctx.experiments
        experiments.run_changed.connect(lambda _: self.fill_networks(keep=True))
        experiments.explain_requested.connect(self.show_run)
        if experiments.explain_target:
            self.show_run(experiments.explain_target)

    # Controls ---------------------------------------------------------------
    def _controls_card(self) -> Card:
        card = Card("Model and input")
        self.network_combo = QComboBox()
        self.network_combo.currentIndexChanged.connect(lambda _: self._network_chosen())
        self.network_combo.setMaximumWidth(380)
        self.scale_combo = QComboBox()
        self.scale_combo.setMaximumWidth(380)
        for key, title in SCALES.items():
            self.scale_combo.addItem(title, key)
        self.scale_combo.setToolTip("Which layers share one colour range: each layer (with its ReLU and "
                                    "pooling), each block, or the whole network")
        self.scale_combo.currentIndexChanged.connect(lambda _: self.recompute())
        self.weights_pill = Pill("Untrained", "warn")
        self.reseed_button = QPushButton(icon("dices"), "New random weights")
        self.reseed_button.clicked.connect(self.reseed)
        weights = QHBoxLayout()
        weights.setSpacing(8)
        weights.addWidget(self.weights_pill)
        weights.addWidget(self.reseed_button)
        weights.addStretch(1)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.addRow("Network", self.network_combo)
        form.addRow("Weights", weights)
        form.addRow("Colour scale", self.scale_combo)
        card.add(form)
        self.weights_note = label("", "CardCaption", wrap=True)
        card.add(self.weights_note)

        samples = QWidget()
        box = QVBoxLayout(samples)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(label("Samples", "CardCaption"))
        self.samples = SampleStrip()
        self.samples.chosen.connect(self.use_sample)
        box.addWidget(self.samples)
        self.input_line = label("", "Mono", wrap=True)
        box.addWidget(self.input_line)
        self.invert_box = QCheckBox("Invert opened images (dark strokes on white paper)")
        box.addWidget(self.invert_box)
        box.addStretch(1)

        self.draw_panel = QWidget()
        draw_box = QVBoxLayout(self.draw_panel)
        draw_box.setContentsMargins(0, 0, 0, 0)
        draw_box.addWidget(label("Draw a digit", "CardCaption"))
        self.pad = DrawPad()
        self.pad.drawn.connect(self.use_drawing)
        draw_box.addWidget(self.pad)
        clear = QPushButton(icon("eraser"), "Clear")
        clear.clicked.connect(self.pad.clear)
        draw_box.addWidget(clear, 0, Qt.AlignmentFlag.AlignLeft)
        card.add(ResponsiveRow([(samples, 1), (self.draw_panel, 0)], breakpoint=560))
        self.error_line = label("", "Body", wrap=True)
        self.error_line.hide()
        card.add(self.error_line)
        return card

    # State ------------------------------------------------------------------
    def class_names(self, candidate: Preset) -> tuple[str, ...]:
        try:
            return tuple(self.ctx.catalog.datasets.get(candidate.dataset_id).class_names)
        except KeyError:
            return ()

    def fill_networks(self, *, keep: bool = False) -> None:
        """Trained TinyVGG runs first (newest first), then the untrained presets."""
        current = self.network_combo.currentData() if keep else None
        self.network_combo.blockSignals(True)
        self.network_combo.clear()
        for view in self.ctx.experiments.explainable():
            meta = trained_meta(view.run_dir)
            accuracy = meta.get("metrics", {}).get("test_acc")
            score = f" · test {accuracy:.1%}" if accuracy is not None else ""
            self.network_combo.addItem(f"Trained · {view.name}{score}", f"run:{view.id}")
        for candidate in PRESETS:
            self.network_combo.addItem(f"Untrained · {candidate.title}", f"preset:{candidate.id}")
        index = self.network_combo.findData(current) if current else 0
        self.network_combo.setCurrentIndex(max(index, 0))
        self.network_combo.blockSignals(False)
        if not keep or index < 0:
            self._network_chosen()

    def show_run(self, run_id: str) -> None:
        self.fill_networks(keep=True)
        index = self.network_combo.findData(f"run:{run_id}")
        if index >= 0:
            self.network_combo.setCurrentIndex(index)

    def _network_chosen(self) -> None:
        data = self.network_combo.currentData() or f"preset:{PRESETS[0].id}"
        kind, _, key = data.partition(":")
        if kind == "run":
            self.use_trained(key)
        else:
            self.set_preset(key)

    def use_trained(self, run_id: str) -> None:
        view = self.ctx.experiments.view(run_id)
        meta = trained_meta(view.run_dir)
        self.trained_run = run_id
        self.preset = preset_for(meta)
        architecture = tiny_vgg(in_channels=meta["in_channels"], size=meta["size"],
                                class_names=tuple(meta["class_names"]), hidden=meta.get("hidden", 10))
        self.net = ExplainerNet.load_npz(architecture, view.run_dir / EXPLAINER_WEIGHTS)
        accuracy = meta.get("metrics", {}).get("test_acc")
        self.weights_pill.setText("Trained")
        self.weights_pill.set_tone("ok")
        self.weights_pill.setToolTip(f"Weights from run {run_id}")
        self.reseed_button.setEnabled(False)
        folded = (" The training normalisation is folded into conv_1_1, so the network reads pixel values "
                  "0 … 1 directly." if meta.get("normalization_folded") else "")
        score = f": test accuracy {accuracy:.2%}." if accuracy is not None else "."
        self.weights_note.setText(f"Trained in “{view.name}” (run {run_id}){score}{folded}")
        self._load_samples()

    def set_preset(self, preset_id: str) -> None:
        self.trained_run = None
        self.preset = preset(preset_id)
        self._build_net()
        self._load_samples()

    def _load_samples(self) -> None:
        self.sample_list = self.preset.samples()
        self.samples.set_samples(self.sample_list)
        self.draw_panel.setVisible(self.preset.channels == 1)
        self.selection = (0, 0)
        self.use_sample(0)

    def reseed(self) -> None:
        if self.trained_run is not None:
            return
        self.seed += 1
        self._build_net()
        self.recompute()

    def _build_net(self) -> None:
        architecture = self.preset.architecture(self.class_names(self.preset))
        self.net = ExplainerNet.untrained(architecture, seed=self.seed)
        self.weights_pill.setText("Untrained")
        self.weights_pill.set_tone("warn")
        self.weights_pill.setToolTip("Random weights: what TinyVGG looks like before training")
        self.reseed_button.setEnabled(True)
        self.weights_note.setText(
            f"Random weights (seed {self.seed}): the arithmetic is real, but the class scores mean "
            "nothing until the network is trained. Train TinyVGG from the Computer Vision lab's "
            "“MNIST · TinyVGG” preset and it appears here as a trained network.")

    def use_sample(self, index: int) -> None:
        sample = self.sample_list[index]
        self.samples.set_current(index)
        self.set_input(sample.image, f"{sample.title} · {sample.credit}")

    def use_drawing(self, values: np.ndarray) -> None:
        self.samples.set_current(None)
        self.set_input(values[None], "your drawing")

    def open_image_file(self, path: str | Path) -> bool:
        try:
            image = load_image(path, self.preset.size, self.preset.channels,
                               invert=self.invert_box.isChecked())
        except ValueError as exc:
            self.error_line.setText(str(exc))
            self.error_line.show()
            return False
        self.samples.set_current(None)
        self.set_input(image, Path(path).name)
        return True

    def _ask_for_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open an image", str(Path.home()), IMAGE_FILTER)
        if path:
            self.open_image_file(path)

    def set_input(self, image: np.ndarray, description: str) -> None:
        self.error_line.hide()
        self.image = np.asarray(image, dtype=np.float64)
        self.input_line.setText(f"input: {description}")
        self.recompute()

    def recompute(self) -> None:
        if self.net is None or self.image is None:
            return
        self.trace = run(self.net, self.image)
        self.limits = map_limits(self.trace, self.scale_combo.currentData())
        self.overview.set_trace(self.trace, self.limits)
        names = self.net.architecture.class_names
        top = self.trace.prediction
        self.prediction.setText(f"top score: {names[top]} · {self.trace.probabilities[top]:.2f}")
        self.prediction.setToolTip("" if self.net.trained else
                                   "Untrained weights: this is not a real prediction yet")
        self.select(*self.selection)

    # Detail -----------------------------------------------------------------
    def select(self, layer: int, node: int) -> None:
        trace = self.trace
        if trace is None:
            return
        self.selection = (layer, node)
        self.overview.set_selected(layer, node)
        if self.detail is not None:
            self.detail_box.removeWidget(self.detail)
            self.detail.hide()
            self.detail.setParent(None)
            self.detail.deleteLater()
        kind = "input" if layer == INPUT else trace.net.architecture.layers[layer].kind
        if layer == INPUT:
            self.detail = InputDetail(trace, node)
            title = "Input"
        else:
            factory = {"conv": ConvDetail, "relu": ReluDetail, "pool": PoolDetail}.get(kind)
            if factory is None:
                self.detail = OutputDetail(trace, node)
                self.detail.class_clicked.connect(lambda unit: self.select(layer, unit))
                title = "Flatten, scores and softmax"
            else:
                self.detail = factory(trace, layer, node, self.limits)
                title = {"conv": "Convolution", "relu": "ReLU", "pool": "Max-pooling"}[kind]
        self.detail_card.title.setText(title)
        channel = "" if kind in ("input", "linear") else f" · channel {node + 1}"
        self.detail_card.caption.setText(describe(trace, layer, node).split(" · ")[0] + channel)
        self.detail_card.caption.show()
        self.detail_facts.setText(describe(trace, layer, node))
        self.detail_box.addWidget(self.detail)
