"""What Model Testing feeds a model: an image (opened, or drawn on a pad), a WAV file, a text, a row of
feature values, or an image with labels / a prompt / a threshold for pretrained vision models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....services.context import AppContext
from ...icons import icon
from ...widgets import FlowGrid, PathLabel, label
from ..explainer.inputs import DrawPad

__all__ = ["IMAGE_FILTER", "InputPanel"]

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif)"
AUDIO_FILTER = "WAV audio (*.wav)"


class InputPanel(QWidget):
    """One panel; ``configure(kind, features)`` shows the controls for that kind of input:
    ``image`` | ``audio`` | ``text`` | ``table`` | ``quantum`` | ``detect`` | ``zero_shot`` | ``caption``."""

    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.kind = "image"
        self.path: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        files = QHBoxLayout()
        self.open_button = QPushButton(icon("folder-open"), "Open image…")
        self.open_button.clicked.connect(self.choose_file)
        files.addWidget(self.open_button)
        self.path_label = PathLabel("")
        files.addWidget(self.path_label, 1)
        layout.addLayout(files)
        self.invert = QCheckBox("Invert (dark strokes on a light background → like MNIST)")
        layout.addWidget(self.invert)
        self.pad_row = QWidget()
        pad_layout = QHBoxLayout(self.pad_row)
        pad_layout.setContentsMargins(0, 0, 0, 0)
        self.pad = DrawPad()
        self.pad.drawn.connect(self._drawn)
        clear = QPushButton(icon("trash-2"), "Clear")
        clear.clicked.connect(self.pad.clear)
        pad_layout.addWidget(self.pad)
        pad_layout.addWidget(label("…or draw here (white on black, like MNIST); the drawing is used as the "
                                   "image.",
                                   "CardCaption", wrap=True), 1)
        pad_layout.addWidget(clear)
        layout.addWidget(self.pad_row)

        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Type or paste a text to classify")
        self.text.setFixedHeight(110)
        layout.addWidget(self.text)
        self.labels = QLineEdit("cat, dog, car, bird, a handwritten digit")
        self.labels.setPlaceholderText("labels, separated by commas")
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Optional start of the caption, e.g. “a photograph of”")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        self.threshold.setPrefix("keep boxes above ")
        self.threshold.setMaximumWidth(220)
        for widget in (self.labels, self.prompt, self.threshold):
            layout.addWidget(widget)
        self.row_holder = QWidget()
        self.row_layout = QVBoxLayout(self.row_holder)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.row_holder)
        self.fields: dict[str, QLineEdit] = {}
        self.configure("image")

    # Setup -----------------------------------------------------------------------------------------
    def configure(self, kind: str, features: list[str] | None = None) -> None:
        self.kind = kind
        images = kind in ("image", "detect", "zero_shot", "caption")
        self.open_button.setText("Open WAV…" if kind in ("audio", "speech") else "Open image…")
        for widget in (self.open_button, self.path_label):
            widget.setVisible(images or kind in ("audio", "speech"))
        self.invert.setVisible(kind == "image")
        self.pad_row.setVisible(kind == "image")
        self.text.setVisible(kind == "text")
        self.labels.setVisible(kind == "zero_shot")
        self.prompt.setVisible(kind == "caption")
        self.threshold.setVisible(kind == "detect")
        self.row_holder.setVisible(kind in ("table", "quantum"))
        if kind in ("table", "quantum"):
            self._fields(features or [])

    def _fields(self, features: list[str]) -> None:
        while self.row_layout.count():
            item = self.row_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.fields = {}
        cells = []
        for name in features:
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            column.addWidget(label(name, "CardCaption"))
            edit = QLineEdit("0")
            edit.textChanged.connect(self.changed)
            column.addWidget(edit)
            self.fields[name] = edit
            cells.append(cell)
        if cells:
            self.row_layout.addWidget(FlowGrid(cells, 150, fewest=2))
        self.row_layout.addWidget(label("Numbers as typed; text columns take their categories (e.g. a "
                                        "species or island). Empty = missing, which the model fills in.",
                                        "CardCaption", wrap=True))

    def set_row(self, row: dict[str, Any]) -> None:
        for name, value in row.items():
            if name in self.fields:
                self.fields[name].setText("" if value is None else str(value))

    # Values ----------------------------------------------------------------------------------------
    def choose_file(self) -> None:
        audio = self.kind in ("audio", "speech")
        title, kinds = ("Open a WAV file", AUDIO_FILTER) if audio else ("Open an image", IMAGE_FILTER)
        path, _ = QFileDialog.getOpenFileName(self, title, str(self.ctx.paths.workspace), kinds)
        if path:
            self.set_path(path)

    def set_path(self, path: str | Path) -> None:
        self.path = str(path)
        self.path_label.setText(self.path)
        self.changed.emit()

    def _drawn(self, values: np.ndarray) -> None:
        from labs.simulation.vision.png import write_png

        if not values.any():
            return
        target = self.ctx.inference.workspace_file("drawing.png")
        write_png(target, (np.clip(values, 0, 1) * 255).astype(np.uint8))
        self.invert.setChecked(False)
        self.set_path(target)

    def row(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, edit in self.fields.items():
            text = edit.text().strip()
            try:
                values[name] = float(text) if text else None
            except ValueError:
                values[name] = text
        return values

    def value(self) -> Any:
        """The input for the worker; ``ValueError`` with what is missing."""
        if self.kind == "text":
            text = self.text.toPlainText().strip()
            if not text:
                raise ValueError("type a text first")
            return text
        if self.kind in ("table", "quantum"):
            return self.row()
        if not self.path:
            raise ValueError("open a file first" + (" or draw on the pad" if self.kind == "image" else ""))
        if self.kind == "image":
            return {"path": self.path, "invert": self.invert.isChecked()}
        return self.path
