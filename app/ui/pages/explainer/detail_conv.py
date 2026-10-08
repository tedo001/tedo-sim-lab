"""Convolution, taken apart (CNN Explainer's convolution view and its "elastic"
intermediate view in one): every input channel is convolved with its own kernel, the
intermediate maps are summed, the bias is added, and that is the output map."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from labs.computer_vision.explainer import Trace, conv_intermediates, conv_step

from ...widgets import ResponsiveRow, label
from .detail_base import MapColumn, NumberGrid, StepView, fmt
from .maps import MapView

__all__ = ["ConvDetail"]

_THUMB = 38


class ConvDetail(StepView):
    def __init__(self, trace: Trace, layer: int, channel: int, limits: dict[int, float],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.trace, self.layer, self.channel = trace, layer, channel
        architecture = trace.net.architecture
        self.spec = architecture.layers[layer]
        self.incoming = trace.layer_input(layer)
        self.output = trace.outputs[layer][channel]
        self.rows, self.columns = self.output.shape
        self.out_limit = limits[layer]
        self.in_limit = None if layer == 0 else limits[layer - 1]
        self.intermediates = conv_intermediates(trace, layer, channel)
        self.source = int(np.argmax(np.abs(self.intermediates).sum(axis=(1, 2))))
        count = self.incoming.shape[0]

        self.layout_.addWidget(label(
            f"Each of the {count} input channels is slid over by its own "
            f"{self.spec.kernel}×{self.spec.kernel} "
            f"kernel (stride {self.spec.stride}, padding {self.spec.padding}). At every position the "
            "window and the kernel are multiplied element by element and summed. That gives one "
            "intermediate map per input channel; their sum plus the bias is this channel's output.",
            "Body", wrap=True))

        self.layout_.addWidget(label("Intermediate maps, one per input channel — click one to follow it",
                                     "CardCaption"))
        strip = QGridLayout()
        strip.setSpacing(6)
        self.thumbs: list[MapView] = []
        for c in range(count):
            thumb = MapView(_THUMB, clickable=True)
            thumb.set_map(self.intermediates[c], self.out_limit)
            thumb.setToolTip(f"input channel {c + 1} ⊛ kernel {c + 1}")
            thumb.clicked.connect(lambda c=c: self.set_source(c))
            strip.addWidget(thumb, 0, c)
            self.thumbs.append(thumb)
        bias = label(f"+ bias {fmt(float(trace.net.bias(self.spec.name)[channel]))}", "Mono")
        strip.addWidget(bias, 0, count)
        strip.setColumnStretch(count + 1, 1)
        holder = QWidget()
        holder.setLayout(strip)
        self.layout_.addWidget(holder)

        self.input_column = MapColumn("")
        self.output_column = MapColumn(f"output · {self.spec.name} channel {channel + 1} · "
                                       f"{self.rows}×{self.columns}")
        self.output_column.view.set_map(self.output, self.out_limit)
        for view in (self.input_column.view, self.output_column.view):
            view.hovered.connect(self._hovered_on(view))

        math = QWidget()
        math_box = QVBoxLayout(math)
        math_box.setContentsMargins(0, 0, 0, 0)
        math_box.setSpacing(4)
        self.math_caption = label("", "CardCaption")
        grids = QHBoxLayout()
        grids.setSpacing(6)
        self.window_grid, self.kernel_grid, self.partial_grid = NumberGrid(), NumberGrid(), NumberGrid()
        for widget in (self.window_grid, label("×", "BodyStrong"), self.kernel_grid,
                       label("=", "BodyStrong"), self.partial_grid):
            grids.addWidget(widget)
        grids.addStretch(1)
        math_box.addWidget(self.math_caption)
        math_box.addLayout(grids)
        math_box.addStretch(1)
        self.layout_.addWidget(ResponsiveRow([(self.input_column, 0), (math, 1), (self.output_column, 0)],
                                             breakpoint=640))

        self.sum_text = label("", "Mono", wrap=True, selectable=True)
        self.layout_.addWidget(self.sum_text)
        self.layout_.addLayout(self.controls("Hover a map to move the kernel; the animation stops "
                                             "until you press Play."))
        self.set_source(self.source)
        self.go_to(0, 0)

    def _hovered_on(self, view: MapView):
        def hovered(y: int, x: int) -> None:
            if view is self.output_column.view:
                self.hover(y, x)
            else:  # a cell of the input: show the window whose top-left is nearest
                stride, padding = self.spec.stride, self.spec.padding
                self.hover((y + padding) // stride, (x + padding) // stride)
        return hovered

    def set_source(self, source: int) -> None:
        self.source = source
        for c, thumb in enumerate(self.thumbs):
            thumb.set_selected(c == source)
        kind = "image" if self.layer == 0 else self.trace.net.architecture.layers[self.layer - 1].name
        values = self.incoming[source]
        self.input_column.caption.setText(f"input · {kind} channel {source + 1} · "
                                          f"{values.shape[0]}×{values.shape[1]}")
        self.input_column.view.set_map(values, self.in_limit)
        self.show_step(*self.position)

    def show_step(self, y: int, x: int) -> None:
        step = conv_step(self.trace, self.layer, self.channel, y, x)
        k = self.spec.kernel
        self.input_column.view.set_window((step.origin[0], step.origin[1], k, k))
        self.output_column.view.set_marker((y, x))
        self.window_grid.set_values(step.windows[self.source], self.in_limit)
        kernel_limit = float(np.abs(step.kernels).max()) or 1.0
        self.kernel_grid.set_values(step.kernels[self.source], kernel_limit)
        self.partial_grid.set_values(np.array([[step.partials[self.source]]]), self.out_limit)
        self.math_caption.setText(f"window × kernel {self.source + 1}, summed · output cell ({y}, {x})")
        terms = " + ".join(f"[{fmt(p)}]" if c == self.source else fmt(p)
                           for c, p in enumerate(step.partials))
        self.sum_text.setText(f"output[{y}, {x}] = {terms} + bias {fmt(step.bias)} = {fmt(step.value)}")
