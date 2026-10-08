"""PyTorch model builders for the vision lab, referenced from ``configs/models/vision.yaml``."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .explainer.network import Architecture, tiny_vgg

__all__ = ["build_tiny_vgg", "to_torch"]


def to_torch(architecture: Architecture) -> Any:
    """An ``nn.Sequential`` whose layers carry the architecture's names, so its ``state_dict``
    loads straight into :class:`~labs.computer_vision.explainer.ExplainerNet`."""
    from torch import nn

    modules: OrderedDict[str, nn.Module] = OrderedDict()
    for index, layer in enumerate(architecture.layers):
        incoming = architecture.input_of(index)
        if layer.kind == "conv":
            modules[layer.name] = nn.Conv2d(incoming[0], layer.channels, layer.kernel,
                                            stride=layer.stride, padding=layer.padding)
        elif layer.kind == "relu":
            modules[layer.name] = nn.ReLU()
        elif layer.kind == "pool":
            modules[layer.name] = nn.MaxPool2d(layer.kernel, stride=layer.stride)
        elif layer.kind == "flatten":
            modules[layer.name] = nn.Flatten()
        else:
            modules[layer.name] = nn.Linear(incoming[0], layer.channels)
    return nn.Sequential(modules)


def build_tiny_vgg(*, num_classes: int, in_channels: int, input_size: tuple[int, int],
                   pretrained: str | None = None, hidden: int = 10, **_: Any) -> Any:
    """CNN Explainer's TinyVGG for square inputs of at least 16×16. No pretrained weights."""
    if pretrained:
        raise ValueError("TinyVGG has no pretrained weights; train it in the lab")
    height, width = input_size
    if height != width:
        raise ValueError(f"TinyVGG here takes square inputs, got {height}×{width}")
    names = tuple(f"class {i}" for i in range(num_classes))
    return to_torch(tiny_vgg(in_channels=in_channels, size=height, class_names=names, hidden=hidden))
