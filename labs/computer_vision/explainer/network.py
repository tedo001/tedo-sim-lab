"""A small sequential CNN described layer by layer, with its weights in numpy.

Ported from the network model of CNN Explainer (Wang et al., 2020; MIT licence, see
``LICENSE-cnn-explainer.txt``) and changed to follow PyTorch's conventions, so a
model trained in the lab is explained exactly as it computes:

* tensors are channels × height × width;
* a convolution is a cross-correlation with weights shaped out × in × k × k;
* flattening goes channel by channel, then row by row;
* parameter names are a ``state_dict``'s (``conv_1_1.weight``, ``output.bias``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["Architecture", "ArchitectureError", "ExplainerNet", "LayerSpec", "conv_output_size",
           "tiny_vgg"]

Kind = Literal["conv", "relu", "pool", "flatten", "linear"]
Shape = tuple[int, ...]


class ArchitectureError(ValueError):
    """The layers do not fit together, or the weights do not fit the layers."""


def conv_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """⌊(n + 2p − k) / s⌋ + 1, PyTorch's rule for convolution and pooling."""
    span = size + 2 * padding - kernel
    if kernel < 1 or stride < 1 or padding < 0:
        raise ArchitectureError("kernel and stride must be at least 1, padding at least 0")
    if span < 0:
        raise ArchitectureError(f"a {kernel}×{kernel} kernel does not fit a {size}×{size} input "
                                f"with padding {padding}")
    return span // stride + 1


@dataclass(frozen=True)
class LayerSpec:
    name: str
    kind: Kind
    #: conv: output channels; linear: output features.
    channels: int = 0
    kernel: int = 0
    stride: int = 1
    padding: int = 0

    @property
    def has_weights(self) -> bool:
        return self.kind in ("conv", "linear")

    @property
    def is_map(self) -> bool:
        """Produces a stack of 2-D activation maps (conv, ReLU, pooling)."""
        return self.kind in ("conv", "relu", "pool")


@dataclass(frozen=True)
class Architecture:
    """Input shape, layers and class names. Shapes are checked on creation."""

    name: str
    input_shape: tuple[int, int, int]
    layers: tuple[LayerSpec, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise ArchitectureError("layer names must be unique")
        if not self.layers or self.layers[-1].kind != "linear":
            raise ArchitectureError("the last layer must be linear (it produces the class scores)")
        if self.shapes[-1] != (len(self.class_names),):
            raise ArchitectureError(f"the output layer has {self.shapes[-1][0]} units but there are "
                                    f"{len(self.class_names)} class names")

    @cached_property
    def shapes(self) -> tuple[Shape, ...]:
        """The output shape of every layer, in order."""
        shapes: list[Shape] = []
        shape: Shape = self.input_shape
        for layer in self.layers:
            shape = _output_shape(layer, shape)
            shapes.append(shape)
        return tuple(shapes)

    def input_of(self, index: int) -> Shape:
        """The shape layer ``index`` receives."""
        return self.input_shape if index == 0 else self.shapes[index - 1]

    @cached_property
    def param_shapes(self) -> dict[str, Shape]:
        """``state_dict``-style name → shape, for every weighted layer."""
        params: dict[str, Shape] = {}
        for index, layer in enumerate(self.layers):
            incoming = self.input_of(index)
            if layer.kind == "conv":
                params[f"{layer.name}.weight"] = (layer.channels, incoming[0], layer.kernel, layer.kernel)
            elif layer.kind == "linear":
                params[f"{layer.name}.weight"] = (layer.channels, incoming[0])
            else:
                continue
            params[f"{layer.name}.bias"] = (layer.channels,)
        return params

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def index_of(self, name: str) -> int:
        for index, layer in enumerate(self.layers):
            if layer.name == name:
                return index
        raise KeyError(name)


def _output_shape(layer: LayerSpec, shape: Shape) -> Shape:
    if layer.kind in ("conv", "pool"):
        if len(shape) != 3:
            raise ArchitectureError(f"{layer.name}: needs a stack of maps, got a vector")
        channels = layer.channels if layer.kind == "conv" else shape[0]
        stride = layer.stride
        try:
            height = conv_output_size(shape[1], layer.kernel, stride, layer.padding)
            width = conv_output_size(shape[2], layer.kernel, stride, layer.padding)
        except ArchitectureError as exc:
            raise ArchitectureError(f"{layer.name}: {exc}") from None
        if layer.kind == "conv" and channels < 1:
            raise ArchitectureError(f"{layer.name}: needs at least one output channel")
        return channels, height, width
    if layer.kind == "relu":
        return shape
    if layer.kind == "flatten":
        return (int(np.prod(shape)),)
    if layer.kind == "linear":
        if len(shape) != 1:
            raise ArchitectureError(f"{layer.name}: flatten the maps before a linear layer")
        return (layer.channels,)
    raise ArchitectureError(f"{layer.name}: unknown layer kind {layer.kind!r}")


def tiny_vgg(*, in_channels: int = 3, size: int = 64, class_names: tuple[str, ...],
             hidden: int = 10, name: str = "TinyVGG") -> Architecture:
    """CNN Explainer's network: two blocks of (3×3 conv, ReLU, 3×3 conv, ReLU, 2×2 max-pool),
    then flatten and one linear layer. Needs inputs of at least 16×16."""
    def conv(layer_name: str) -> LayerSpec:
        return LayerSpec(layer_name, "conv", channels=hidden, kernel=3)

    layers = (
        conv("conv_1_1"), LayerSpec("relu_1_1", "relu"), conv("conv_1_2"), LayerSpec("relu_1_2", "relu"),
        LayerSpec("max_pool_1", "pool", kernel=2, stride=2),
        conv("conv_2_1"), LayerSpec("relu_2_1", "relu"), conv("conv_2_2"), LayerSpec("relu_2_2", "relu"),
        LayerSpec("max_pool_2", "pool", kernel=2, stride=2),
        LayerSpec("flatten", "flatten"),
        LayerSpec("output", "linear", channels=len(class_names)),
    )
    return Architecture(name, (in_channels, size, size), layers, tuple(class_names))


@dataclass
class ExplainerNet:
    """An architecture plus its parameters (float64 arrays keyed like a ``state_dict``)."""

    architecture: Architecture
    params: dict[str, np.ndarray]
    trained: bool = False
    #: Where the weights came from: "random, seed 0", a run id, a file.
    source: str = ""

    def __post_init__(self) -> None:
        expected = self.architecture.param_shapes
        missing = sorted(set(expected) - set(self.params))
        extra = sorted(set(self.params) - set(expected))
        if missing or extra:
            raise ArchitectureError(f"weights do not match {self.architecture.name}: "
                                    f"missing {missing or 'none'}, unexpected {extra or 'none'}")
        params = {}
        for name, shape in expected.items():
            array = np.asarray(self.params[name], dtype=np.float64)
            if array.shape != shape:
                raise ArchitectureError(f"{name} has shape {array.shape}, expected {shape}")
            params[name] = array
        self.params = params

    @classmethod
    def untrained(cls, architecture: Architecture, seed: int = 0) -> ExplainerNet:
        """PyTorch's default initialisation: weights and biases drawn from U(−1/√fan_in, 1/√fan_in)."""
        rng = np.random.default_rng(seed)
        params: dict[str, np.ndarray] = {}
        for name, shape in architecture.param_shapes.items():
            weight_shape = architecture.param_shapes[name.rsplit(".", 1)[0] + ".weight"]
            bound = 1.0 / np.sqrt(int(np.prod(weight_shape[1:])))
            params[name] = rng.uniform(-bound, bound, size=shape)
        return cls(architecture, params, trained=False, source=f"random, seed {seed}")

    def weight(self, layer: str) -> np.ndarray:
        return self.params[f"{layer}.weight"]

    def bias(self, layer: str) -> np.ndarray:
        return self.params[f"{layer}.bias"]

    def save_npz(self, path: Path) -> None:
        np.savez(path, **{name: value.astype(np.float32) for name, value in self.params.items()})

    @classmethod
    def load_npz(cls, architecture: Architecture, path: Path, *, trained: bool = True) -> ExplainerNet:
        with np.load(path) as data:
            params = {name: data[name] for name in data.files}
        return cls(architecture, params, trained=trained, source=str(path))
