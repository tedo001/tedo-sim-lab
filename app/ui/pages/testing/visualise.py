"""Inside a trained image or spectrogram network, for one input, the way the CNN Explainer shows
TinyVGG but for any network the lab trains: the input as the network sees it, where it looked
(Grad-CAM) and which pixels mattered (saliency), then every layer's feature maps, strongest
channels first, coloured on the diverging activation scale (negative · zero · positive)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...theme.tokens import COLORS
from ...widgets import Card, FlowGrid, label
from ...widgets.plots import BarList

__all__ = ["LayerMaps", "colour_maps", "decode"]

TILE = 72


def decode(data: str, shape: list[int]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.uint8).reshape(shape)


def _rgb(name: str) -> np.ndarray:
    value = COLORS[name].lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def colour_maps(scaled: np.ndarray, low: float, high: float) -> np.ndarray:
    """uint8 maps (0..255 over low..high) → RGB on the heat scale: heat_neg · heat_zero · heat_pos,
    symmetric around zero so a sign reads the same in every map."""
    values = low + scaled.astype(float) / 255.0 * (high - low)
    reach = max(abs(low), abs(high)) or 1.0
    t = np.clip(values / reach, -1, 1)[..., None]
    zero, neg, pos = _rgb("heat_zero"), _rgb("heat_neg"), _rgb("heat_pos")
    rgb = np.where(t < 0, zero + (neg - zero) * -t, zero + (pos - zero) * t)
    return rgb.round().astype(np.uint8)


def _pixmap(array: np.ndarray, side: int = TILE) -> QPixmap:
    data = np.ascontiguousarray(array)
    fmt = QImage.Format.Format_RGB888 if data.ndim == 3 else QImage.Format.Format_Grayscale8
    image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], fmt).copy()
    return QPixmap.fromImage(image).scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.FastTransformation)


def _tile(array: np.ndarray, caption: str, side: int = TILE) -> QWidget:
    tile = QWidget()
    column = QVBoxLayout(tile)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    picture = QLabel()
    picture.setPixmap(_pixmap(array, side))
    picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(picture)
    column.addWidget(label(caption, "Mono"), 0, Qt.AlignmentFlag.AlignHCenter)
    return tile


def _overlay(base: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """The input in grey with the heat map laid over it in the positive activation colour."""
    grey = base if base.ndim == 2 else base.mean(axis=2)
    grey = np.repeat(grey[..., None].astype(float), 3, axis=2) * 0.55
    weight = 0.65 * (heat.astype(float) / 255.0)[..., None]  # the input always shows through
    return (grey * (1 - weight) + _rgb("heat_pos") * weight).clip(0, 255).astype(np.uint8)


class LayerMaps(QWidget):
    """Filled from a ``visualize`` answer of the inference worker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(12)

    def clear(self) -> None:
        while self.column.count():
            widget = self.column.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def show_result(self, result: dict[str, Any]) -> None:
        self.clear()
        classes, probs = result.get("classes", []), result.get("probabilities", [])
        target = int(result.get("target", 0))
        top = Card("Where it looked", f"explaining “{classes[target] if target < len(classes) else target}”")
        row = QHBoxLayout()
        shown = decode(result["input"]["data"], result["input"]["shape"])
        tiles = [_tile(shown, "input", 120)]
        if "gradcam" in result:
            cam = decode(result["gradcam"]["data"], result["gradcam"]["shape"])
            base = shown if shown.shape[:2] == cam.shape else np.zeros(cam.shape, dtype=np.uint8)
            tiles.append(_tile(_overlay(base, cam), "Grad-CAM", 120))
        if "saliency" in result:
            tiles.append(_tile(decode(result["saliency"]["data"], result["saliency"]["shape"]), "saliency",
                               120))
        row.addWidget(FlowGrid(tiles, 128, fewest=1), 1)
        ranking = BarList()
        order = np.argsort(-np.asarray(probs))[:10] if probs else []
        ranking.set_rows([[classes[i] if i < len(classes) else str(i), probs[i]] for i in order])
        side = QVBoxLayout()
        counts = f"{result.get('parameters', 0):,} parameters · {len(result.get('layers', []))} layers"
        side.addWidget(label(counts, "CardCaption"))
        side.addWidget(ranking)
        side.addStretch(1)
        row.addLayout(side, 1)
        top.add(row)
        top.add(label("Grad-CAM weighs the last convolution's maps by how much they raise this class's "
                      "score; "
                      "saliency is how strongly each input pixel changes it.", "CardCaption", wrap=True))
        self.column.addWidget(top)
        for layer in result.get("layers", []):
            self.column.addWidget(self._layer(layer))

    def _layer(self, layer: dict[str, Any]) -> Card:
        shape = "×".join(str(v) for v in layer["shape"])
        card = Card(f"{layer['name']} · {layer['kind']}", shape)
        if "maps" in layer:
            maps = decode(layer["maps"], layer["map_shape"])
            coloured = colour_maps(maps, float(layer["low"]), float(layer["high"]))
            channels = layer.get("strongest") or list(range(len(maps)))  # map k is channel channels[k]
            tiles = [_tile(coloured[k], f"#{channel}") for k, channel in enumerate(channels[:len(maps)])]
            card.add(FlowGrid(tiles, TILE + 6, fewest=2))
            more = int(layer.get("channels", len(maps))) - len(maps)
            caption = f"values {layer['low']:.2f} … {layer['high']:.2f}" + (f" · {more} more channels"
                                                                            if more > 0
                                                                              else "")
            card.add(label(caption, "CardCaption"))
        else:
            values = layer.get("values", [])
            bars = BarList()
            order = np.argsort(-np.asarray(values))[:8] if values else []
            bars.set_rows([[f"unit {i}", values[i]] for i in order])
            card.add(bars)
            card.add(label(f"the 8 strongest of {layer.get('size', len(values))} values", "CardCaption"))
        return card
