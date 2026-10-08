"""Run an image through an :class:`ExplainerNet`, keeping every intermediate result,
and take any single output value apart into the arithmetic that produced it.

The step functions are what CNN Explainer's "interactive formula" views show: the
kernel window and its element-wise products for a convolution, max(0, x) for ReLU,
the window maximum for pooling, and the softmax fraction for a class score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .network import ExplainerNet, LayerSpec

__all__ = ["ConvStep", "PoolStep", "SoftmaxTerms", "Trace", "conv2d", "conv_intermediates",
           "conv_step", "linear_contributions", "max_pool", "pool_step", "run", "softmax",
           "softmax_terms"]


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1,
           padding: int = 0) -> np.ndarray:
    """(C, H, W) ⊛ (O, C, k, k) + (O,) → (O, H', W'), as ``torch.nn.functional.conv2d``."""
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    k = weight.shape[-1]
    windows = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::stride, ::stride]
    return np.einsum("chwij,ocij->ohw", windows, weight, optimize=True) + bias[:, None, None]


def max_pool(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    windows = sliding_window_view(x, (kernel, kernel), axis=(1, 2))[:, ::stride, ::stride]
    return windows.max(axis=(-2, -1))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def _apply(layer: LayerSpec, net: ExplainerNet, x: np.ndarray) -> np.ndarray:
    if layer.kind == "conv":
        return conv2d(x, net.weight(layer.name), net.bias(layer.name), layer.stride, layer.padding)
    if layer.kind == "relu":
        return np.maximum(x, 0.0)
    if layer.kind == "pool":
        return max_pool(x, layer.kernel, layer.stride)
    if layer.kind == "flatten":
        return x.reshape(-1)
    return net.weight(layer.name) @ x + net.bias(layer.name)


@dataclass(frozen=True)
class Trace:
    """Everything the network computed for one input."""

    net: ExplainerNet
    #: (C, H, W), values 0..1.
    input: np.ndarray
    #: One array per layer: (C, H, W) maps, or vectors after flatten.
    outputs: tuple[np.ndarray, ...]

    @property
    def logits(self) -> np.ndarray:
        return self.outputs[-1]

    @property
    def probabilities(self) -> np.ndarray:
        return softmax(self.logits)

    @property
    def prediction(self) -> int:
        return int(np.argmax(self.logits))

    def layer_input(self, index: int) -> np.ndarray:
        """What layer ``index`` received."""
        return self.input if index == 0 else self.outputs[index - 1]


def run(net: ExplainerNet, image: np.ndarray) -> Trace:
    """Forward pass. ``image`` is (C, H, W) in the network's input shape."""
    x = np.asarray(image, dtype=np.float64)
    if x.shape != net.architecture.input_shape:
        raise ValueError(f"input has shape {x.shape}, the network expects "
                         f"{net.architecture.input_shape}")
    outputs = []
    for layer in net.architecture.layers:
        x = _apply(layer, net, x)
        outputs.append(x)
    return Trace(net, np.asarray(image, dtype=np.float64), tuple(outputs))


# ── One output value, taken apart ─────────────────────────────────────────────

@dataclass(frozen=True)
class ConvStep:
    """How ``output[channel, y, x]`` of a convolution came about."""

    y: int
    x: int
    #: Top-left corner of the kernel window in the input; negative inside the padding.
    origin: tuple[int, int]
    #: (C_in, k, k): the input values under the window (zeros where it covers padding).
    windows: np.ndarray
    #: (C_in, k, k): the kernel applied to each input channel.
    kernels: np.ndarray
    #: (C_in,): Σ window × kernel for each input channel.
    partials: np.ndarray
    bias: float

    @property
    def value(self) -> float:
        return float(self.partials.sum() + self.bias)


def _window(x: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """``x[:, top:top+size, left:left+size]`` with zeros outside ``x``."""
    channels, height, width = x.shape
    out = np.zeros((channels, size, size))
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + size, height), min(left + size, width)
    if y1 > y0 and x1 > x0:
        out[:, y0 - top:y1 - top, x0 - left:x1 - left] = x[:, y0:y1, x0:x1]
    return out


def conv_step(trace: Trace, index: int, channel: int, y: int, x: int) -> ConvStep:
    layer = trace.net.architecture.layers[index]
    k, stride, padding = layer.kernel, layer.stride, layer.padding
    top, left = y * stride - padding, x * stride - padding
    windows = _window(trace.layer_input(index), top, left, k)
    kernels = trace.net.weight(layer.name)[channel]
    partials = (windows * kernels).sum(axis=(1, 2))
    return ConvStep(y, x, (top, left), windows, kernels, partials,
                    float(trace.net.bias(layer.name)[channel]))


def conv_intermediates(trace: Trace, index: int, channel: int) -> np.ndarray:
    """(C_in, H', W'): each input channel convolved with its own kernel, before the sum and
    the bias. Their sum plus the bias is the output map."""
    layer = trace.net.architecture.layers[index]
    x = trace.layer_input(index)
    weight = trace.net.weight(layer.name)[channel]
    maps = [conv2d(x[c:c + 1], weight[None, c:c + 1], np.zeros(1), layer.stride, layer.padding)[0]
            for c in range(x.shape[0])]
    return np.stack(maps)


@dataclass(frozen=True)
class PoolStep:
    y: int
    x: int
    origin: tuple[int, int]
    #: (k, k): the values under the window.
    window: np.ndarray
    #: Position of the maximum inside the window.
    argmax: tuple[int, int]

    @property
    def value(self) -> float:
        return float(self.window[self.argmax])


def pool_step(trace: Trace, index: int, channel: int, y: int, x: int) -> PoolStep:
    layer = trace.net.architecture.layers[index]
    top, left = y * layer.stride, x * layer.stride
    window = trace.layer_input(index)[channel, top:top + layer.kernel, left:left + layer.kernel]
    position = np.unravel_index(int(np.argmax(window)), window.shape)
    return PoolStep(y, x, (top, left), window.copy(), (int(position[0]), int(position[1])))


@dataclass(frozen=True)
class SoftmaxTerms:
    logits: np.ndarray
    #: exp(z_i) for every class.
    exps: np.ndarray

    @property
    def total(self) -> float:
        return float(self.exps.sum())

    def probability(self, index: int) -> float:
        return float(self.exps[index] / self.total)


def softmax_terms(logits: np.ndarray) -> SoftmaxTerms:
    """The fraction behind each probability. Logits are shifted by their maximum when they
    are large enough to overflow exp(); the probabilities are the same either way."""
    shift = float(logits.max()) if logits.max() > 50 else 0.0
    return SoftmaxTerms(logits - shift, np.exp(logits - shift))


def linear_contributions(trace: Trace, index: int, unit: int) -> tuple[np.ndarray, float]:
    """weight × input for one unit of a linear layer, shaped like the maps that were flattened
    (or like its input vector), plus the bias. Their total is the unit's value."""
    layer = trace.net.architecture.layers[index]
    vector = trace.layer_input(index)
    contributions = trace.net.weight(layer.name)[unit] * vector
    architecture = trace.net.architecture
    if index > 0 and architecture.layers[index - 1].kind == "flatten":
        contributions = contributions.reshape(architecture.input_of(index - 1))
    return contributions, float(trace.net.bias(layer.name)[unit])
