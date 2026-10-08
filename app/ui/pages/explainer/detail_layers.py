"""The ReLU, max-pooling and input views."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from labs.computer_vision.explainer import Trace, pool_step

from ...widgets import ResponsiveRow, label
from .colours import rgb_image
from .detail_base import MapColumn, NumberGrid, StepView, fmt

__all__ = ["InputDetail", "PoolDetail", "ReluDetail"]


class ReluDetail(StepView):
    """max(0, x), one value at a time."""

    def __init__(self, trace: Trace, layer: int, channel: int, limits: dict[int, float],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.before = trace.layer_input(layer)[channel]
        self.after = trace.outputs[layer][channel]
        self.rows, self.columns = self.after.shape
        name = trace.net.architecture.layers[layer].name
        source = trace.net.architecture.layers[layer - 1].name if layer else "input"
        self.layout_.addWidget(label(
            "ReLU keeps positive values and replaces negative ones with zero: max(0, x), value by "
            "value. It is what makes the network non-linear; without it, stacked convolutions would "
            "collapse into a single one. Red (negative) cells turn neutral; blue cells pass unchanged.",
            "Body", wrap=True))
        limit = limits[layer]
        self.left = MapColumn(f"input · {source} channel {channel + 1}")
        self.right = MapColumn(f"output · {name} channel {channel + 1}")
        self.left.view.set_map(self.before, limit)
        self.right.view.set_map(self.after, limit)
        for view in (self.left.view, self.right.view):
            view.hovered.connect(self.hover)
        self.formula = label("", "KvValueMono")
        middle = QWidget()
        box = QVBoxLayout(middle)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(label("ReLU at the outlined cell", "CardCaption"))
        box.addWidget(self.formula)
        box.addStretch(1)
        self.layout_.addWidget(ResponsiveRow([(self.left, 0), (middle, 1), (self.right, 0)], breakpoint=600))
        self.layout_.addLayout(self.controls("Hover a map to pick a cell."))
        self.go_to(0, 0)

    def show_step(self, y: int, x: int) -> None:
        for view in (self.left.view, self.right.view):
            view.set_marker((y, x))
        before, after = float(self.before[y, x]), float(self.after[y, x])
        self.formula.setText(f"max(0, {fmt(before)}) = {fmt(after)}")


class PoolDetail(StepView):
    """The largest value in each window."""

    def __init__(self, trace: Trace, layer: int, channel: int, limits: dict[int, float],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.trace, self.layer, self.channel = trace, layer, channel
        self.spec = trace.net.architecture.layers[layer]
        self.output = trace.outputs[layer][channel]
        self.rows, self.columns = self.output.shape
        k, stride = self.spec.kernel, self.spec.stride
        dropped = 1 - 1 / (k * k) if stride == k else None
        share = f" It keeps one value in {k * k} and drops {dropped:.0%}." if dropped else ""
        self.layout_.addWidget(label(
            f"Max-pooling slides a {k}×{k} window with stride {stride} and keeps the largest value in "
            f"each window, halving the map's width and height here.{share} Fewer values means less "
            "computation in the layers that follow and some tolerance to small shifts.",
            "Body", wrap=True))
        self.limit = limits[layer]
        source = trace.layer_input(layer)[channel]
        previous = trace.net.architecture.layers[layer - 1].name
        self.left = MapColumn(f"input · {previous} channel {channel + 1} · "
                              f"{source.shape[0]}×{source.shape[1]}")
        self.right = MapColumn(f"output · {self.spec.name} channel {channel + 1} · "
                               f"{self.rows}×{self.columns}")
        self.left.view.set_map(source, self.limit)
        self.right.view.set_map(self.output, self.limit)
        self.left.view.hovered.connect(lambda y, x: self.hover(y // stride, x // stride))
        self.right.view.hovered.connect(self.hover)
        self.grid = NumberGrid(46)
        self.formula = label("", "KvValueMono")
        middle = QWidget()
        box = QVBoxLayout(middle)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(label("the window; its maximum is outlined", "CardCaption"))
        row = QHBoxLayout()
        row.addWidget(self.grid)
        row.addStretch(1)
        box.addLayout(row)
        box.addWidget(self.formula)
        box.addStretch(1)
        self.layout_.addWidget(ResponsiveRow([(self.left, 0), (middle, 1), (self.right, 0)], breakpoint=600))
        self.layout_.addLayout(self.controls("Hover a map to move the window."))
        self.go_to(0, 0)

    def show_step(self, y: int, x: int) -> None:
        step = pool_step(self.trace, self.layer, self.channel, y, x)
        k = self.spec.kernel
        self.left.view.set_window((step.origin[0], step.origin[1], k, k))
        self.right.view.set_marker((y, x))
        self.grid.set_values(step.window, self.limit, marked=step.argmax)
        self.formula.setText(f"max({', '.join(fmt(v) for v in step.window.ravel())}) = {fmt(step.value)}")


class InputDetail(QWidget):
    """One channel of the input image."""

    def __init__(self, trace: Trace, channel: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        channels = trace.input.shape[0]
        name = ("red", "green", "blue")[channel] if channels == 3 else "grey"
        layout.addWidget(label(
            f"The input layer is the image itself: {channels} channel{'s' if channels > 1 else ''}, "
            "one number per pixel from 0 (black) to 1 (white). A colour image has a red, a green and "
            "a blue channel; the first convolution looks at all of them at once.", "Body", wrap=True))
        self.image = MapColumn("the image")
        self.image.view.set_image(rgb_image(trace.input))
        self.channel = MapColumn(f"{name} channel · {trace.input.shape[1]}×{trace.input.shape[2]}")
        self.channel.view.set_map(trace.input[channel], None)
        self.value = label("", "Mono")
        self.channel.view.hovered.connect(lambda y, x: self.value.setText(
            f"pixel ({y}, {x}) = {float(trace.input[channel, y, x]):.3f}"))
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self.image)
        row.addWidget(self.channel)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.value)
        stats = trace.input[channel]
        layout.addWidget(label(f"min {stats.min():.3f} · mean {np.mean(stats):.3f} · max {stats.max():.3f}",
                               "Mono"))
