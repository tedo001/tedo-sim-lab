"""A network as a stack of layers, described as data (so it fits in ``experiment.yaml``):
validation, output shape after every layer, parameter counts and the equivalent PyTorch code.
No PyTorch needed here; :mod:`labs.deep_learning.model` builds the real module.

A layer is a mapping with a ``type`` and its settings::

    {type: conv, channels: 16, kernel: 3, stride: 1, padding: 1}
    {type: batchnorm} {type: relu} {type: maxpool, kernel: 2} {type: dropout, p: 0.25}
    {type: flatten} {type: linear, features: 128}

The classifier's last layer (``Linear`` to the number of classes) is added automatically, and
a ``flatten`` is inserted before the first ``linear`` when a stack forgets it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["KINDS", "LayerError", "Step", "plan", "pytorch_code", "starter_stack"]

#: type → (title, settings with defaults, help).
KINDS: dict[str, tuple[str, dict[str, Any], str]] = {
    "conv": ("Conv2d", {"channels": 16, "kernel": 3, "stride": 1, "padding": 1},
             "learns local patterns; channels = how many"),
    "batchnorm": ("BatchNorm2d", {}, "keeps activations in a steady range"),
    "relu": ("ReLU", {}, "keeps positives, zeroes negatives"),
    "maxpool": ("MaxPool2d", {"kernel": 2}, "halves the size (kernel 2) keeping the strongest"),
    "avgpool": ("AdaptiveAvgPool2d", {"size": 1}, "averages each map to size × size"),
    "dropout": ("Dropout", {"p": 0.25}, "randomly silences units while training"),
    "flatten": ("Flatten", {}, "turns maps into one vector"),
    "linear": ("Linear", {"features": 128}, "fully connected layer"),
}


class LayerError(ValueError):
    """A stack that cannot work; the message says which layer and why."""


@dataclass(frozen=True)
class Step:
    index: int
    kind: str
    title: str
    settings: dict[str, Any]
    #: (C, H, W) for maps, (N,) for vectors.
    shape: tuple[int, ...]
    parameters: int
    automatic: bool = False

    def describe(self) -> str:
        inside = ", ".join(f"{k}={v}" for k, v in self.settings.items())
        return f"{self.title}({inside})" if inside else self.title


def starter_stack() -> list[dict[str, Any]]:
    """A sensible first network for small images: two convolution blocks and a hidden layer."""
    return [{"type": "conv", "channels": 16, "kernel": 3, "stride": 1, "padding": 1}, {"type": "batchnorm"},
            {"type": "relu"}, {"type": "maxpool", "kernel": 2},
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 1, "padding": 1}, {"type": "batchnorm"},
            {"type": "relu"}, {"type": "maxpool", "kernel": 2},
            {"type": "flatten"}, {"type": "linear", "features": 64}, {"type": "relu"},
            {"type": "dropout", "p": 0.25}]


def _int(layer: Mapping[str, Any], key: str, index: int, low: int = 1) -> int:
    value = layer.get(key, KINDS[layer["type"]][1][key])
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise LayerError(f"layer {index + 1} ({layer['type']}): {key} must be a whole number "
                         f"of at least {low}")
    return value


def _settings(layer: Mapping[str, Any], index: int) -> dict[str, Any]:
    kind = layer.get("type")
    if kind not in KINDS:
        raise LayerError(f"layer {index + 1}: unknown type {kind!r} (known: {', '.join(KINDS)})")
    unknown = set(layer) - {"type"} - set(KINDS[kind][1])
    if unknown:
        raise LayerError(f"layer {index + 1} ({kind}): unknown setting(s) {', '.join(sorted(unknown))}")
    return {key: layer.get(key, default) for key, default in KINDS[kind][1].items()}


def _auto_flatten(steps: list[Step], shape: tuple[int, ...]) -> tuple[int, ...]:
    steps.append(Step(len(steps), "flatten", "Flatten", {}, (shape[0] * shape[1] * shape[2],), 0, True))
    return steps[-1].shape


def plan(layers: Sequence[Mapping[str, Any]], *, in_channels: int, input_size: tuple[int, int],
         num_classes: int) -> list[Step]:
    """Every layer with its output shape and parameters, plus the automatic flatten and the final
    ``Linear(num_classes)``; raises :class:`LayerError` for a stack that cannot work."""
    shape: tuple[int, ...] = (in_channels, *input_size)
    steps: list[Step] = []
    for index, layer in enumerate(layers):
        settings = _settings(layer, index)
        kind = layer["type"]
        if kind == "linear" and len(shape) == 3:
            shape = _auto_flatten(steps, shape)
        maps = len(shape) == 3
        if kind in ("conv", "batchnorm", "maxpool", "avgpool") and not maps:
            raise LayerError(f"layer {index + 1} ({kind}) needs image maps, but the input here is already a "
                             "vector: move it before Flatten/Linear")
        parameters = 0
        if kind == "conv":
            channels, kernel = _int(layer, "channels", index), _int(layer, "kernel", index)
            stride, padding = _int(layer, "stride", index), _int(layer, "padding", index, low=0)
            height = (shape[1] + 2 * padding - kernel) // stride + 1
            width = (shape[2] + 2 * padding - kernel) // stride + 1
            if height < 1 or width < 1:
                raise LayerError(f"layer {index + 1} (conv): a {kernel}×{kernel} kernel does not fit "
                                 f"{shape[1]}×{shape[2]} maps")
            parameters = channels * shape[0] * kernel * kernel + channels
            shape = (channels, height, width)
        elif kind == "batchnorm":
            parameters = 2 * shape[0]
        elif kind == "maxpool":
            kernel = _int(layer, "kernel", index)
            if shape[1] < kernel or shape[2] < kernel:
                raise LayerError(f"layer {index + 1} (maxpool): {shape[1]}×{shape[2]} maps are smaller "
                                 f"than the {kernel}×{kernel} window")
            shape = (shape[0], shape[1] // kernel, shape[2] // kernel)
        elif kind == "avgpool":
            size = _int(layer, "size", index)
            shape = (shape[0], size, size)
        elif kind == "dropout":
            p = settings["p"]
            if not isinstance(p, (int, float)) or not 0 <= p < 1:
                raise LayerError(f"layer {index + 1} (dropout): p must be between 0 and 1")
        elif kind == "flatten":
            shape = (shape[0] * shape[1] * shape[2],) if maps else shape
        elif kind == "linear":
            features = _int(layer, "features", index)
            parameters = shape[0] * features + features
            shape = (features,)
        steps.append(Step(len(steps), kind, KINDS[kind][0], settings, shape, parameters))
    if len(shape) == 3:
        shape = _auto_flatten(steps, shape)
    steps.append(Step(len(steps), "linear", "Linear", {"features": num_classes}, (num_classes,),
                      shape[0] * num_classes + num_classes, True))
    return steps


def pytorch_code(steps: Sequence[Step], *, in_channels: int, name: str = "LayerStack") -> str:
    """The stack as a self-contained PyTorch module (what the lab builds, written out)."""
    lines = ["import torch.nn as nn", "", "", f"class {name}(nn.Module):", "    def __init__(self):",
             "        super().__init__()", "        self.layers = nn.Sequential("]
    channels, features = in_channels, None
    for step in steps:
        s = step.settings
        if step.kind == "conv":
            code = (f"nn.Conv2d({channels}, {s['channels']}, kernel_size={s['kernel']}, "
                    f"stride={s['stride']}, padding={s['padding']})")
            channels = s["channels"]
        elif step.kind == "batchnorm":
            code = f"nn.BatchNorm2d({channels})"
        elif step.kind == "relu":
            code = "nn.ReLU()"
        elif step.kind == "maxpool":
            code = f"nn.MaxPool2d({s['kernel']})"
        elif step.kind == "avgpool":
            code = f"nn.AdaptiveAvgPool2d({s['size']})"
        elif step.kind == "dropout":
            code = f"nn.Dropout(p={s['p']})"
        elif step.kind == "flatten":
            code, features = "nn.Flatten()", step.shape[0]
        else:
            code = f"nn.Linear({features}, {s['features']})"
            features = s["features"]
        note = "  # added automatically" if step.automatic else ""
        lines.append(f"            {code},{note}  # → {' × '.join(map(str, step.shape))}")
    lines += ["        )", "", "    def forward(self, x):", "        return self.layers(x)", ""]
    return "\n".join(lines)
