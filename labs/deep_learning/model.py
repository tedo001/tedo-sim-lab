"""Builds the PyTorch module for a layer stack drawn on the Deep Learning page (model card
``layer_stack``); the stack travels in ``experiment.yaml`` as ``model.params.layers``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .layers import Step, plan, starter_stack

__all__ = ["build_layer_stack", "module_for"]


def _module(step: Step, channels: int, features: int | None) -> Any:
    from torch import nn

    s = step.settings
    if step.kind == "conv":
        return nn.Conv2d(channels, s["channels"], s["kernel"], stride=s["stride"], padding=s["padding"])
    if step.kind == "batchnorm":
        return nn.BatchNorm2d(channels)
    if step.kind == "relu":
        return nn.ReLU()
    if step.kind == "maxpool":
        return nn.MaxPool2d(s["kernel"])
    if step.kind == "avgpool":
        return nn.AdaptiveAvgPool2d(s["size"])
    if step.kind == "dropout":
        return nn.Dropout(float(s["p"]))
    if step.kind == "flatten":
        return nn.Flatten()
    return nn.Linear(features, s["features"])


def module_for(steps: Sequence[Step], *, in_channels: int) -> Any:
    """An ``nn.Sequential`` with one module per planned step."""
    from torch import nn

    modules, channels, features = [], in_channels, None
    for step in steps:
        modules.append(_module(step, channels, features))
        if step.kind == "conv":
            channels = step.settings["channels"]
        if len(step.shape) == 1:
            features = step.shape[0]
    return nn.Sequential(*modules)


def build_layer_stack(*, num_classes: int, in_channels: int, input_size: tuple[int, int],
                      pretrained: str | None = None,
                      layers: Sequence[Mapping[str, Any]] | None = None, **_: Any) -> Any:
    """The network described by ``layers`` (default: :func:`starter_stack`) for this input,
    ending in ``Linear(num_classes)``. Raises ``ValueError`` for a stack that cannot work."""
    if pretrained:
        raise ValueError("a layer stack has no pretrained weights; train it in the lab")
    steps = plan(list(layers) if layers is not None else starter_stack(), in_channels=in_channels,
                 input_size=tuple(input_size), num_classes=num_classes)
    return module_for(steps, in_channels=in_channels)
