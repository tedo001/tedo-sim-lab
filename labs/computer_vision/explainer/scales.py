"""Colour ranges for activation maps, as in CNN Explainer.

Every convolution sets a symmetric range ±max|value| that its ReLU and pooling
layers reuse, so the same value has the same colour along conv → ReLU → pool and
ReLU's effect (negatives become zero) is visible. The range can be widened to the
block (everything up to and including a pooling layer) or to the whole network,
which makes layers comparable but faint layers fainter.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .forward import Trace

__all__ = ["SCALES", "Scale", "map_limits"]

Scale = Literal["layer", "block", "network"]
#: Scale → label for the interface.
SCALES: dict[Scale, str] = {"layer": "Per layer", "block": "Per block", "network": "Whole network"}

_FLOOR = 1e-6


def map_limits(trace: Trace, scale: Scale = "layer") -> dict[int, float]:
    """Layer index → the half-width of its colour range (values span −limit … +limit), for
    every layer that produces activation maps."""
    layers = trace.net.architecture.layers
    local: dict[int, float] = {}
    blocks: dict[int, int] = {}
    current = _FLOOR
    block = 0
    for index, layer in enumerate(layers):
        if not layer.is_map:
            continue
        if layer.kind == "conv" or current == _FLOOR:
            current = max(float(np.abs(trace.outputs[index]).max()), _FLOOR)
        local[index] = current
        blocks[index] = block
        if layer.kind == "pool":
            block += 1
    if scale == "layer":
        return local
    if scale == "network":
        widest = max(local.values(), default=_FLOOR)
        return dict.fromkeys(local, widest)
    per_block: dict[int, float] = {}
    for index, number in blocks.items():
        per_block[number] = max(per_block.get(number, _FLOOR), local[index])
    return {index: per_block[number] for index, number in blocks.items()}
