"""PyTorch model builders for the vision lab, referenced from ``configs/models/vision.yaml``.

Every builder takes ``num_classes``, ``in_channels`` and ``input_size`` and adapts to them,
so one model card works for MNIST (1 × 28 × 28) and CIFAR-10 (3 × 32 × 32) alike.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .explainer.network import Architecture, tiny_vgg

__all__ = ["build_lenet5", "build_resnet18", "build_simple_cnn", "build_tiny_vgg", "to_torch"]


def _flat_features(features: Any, in_channels: int, input_size: tuple[int, int]) -> int:
    """How many values ``features`` produces for one input, found by running it once."""
    import torch

    with torch.no_grad():
        return int(features(torch.zeros(1, in_channels, *input_size)).numel())


def _no_pretrained(name: str, pretrained: str | None) -> None:
    if pretrained:
        raise ValueError(f"{name} has no pretrained weights; train it in the lab")


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
    _no_pretrained("TinyVGG", pretrained)
    height, width = input_size
    if height != width:
        raise ValueError(f"TinyVGG here takes square inputs, got {height}×{width}")
    names = tuple(f"class {i}" for i in range(num_classes))
    return to_torch(tiny_vgg(in_channels=in_channels, size=height, class_names=names, hidden=hidden))


def build_simple_cnn(*, num_classes: int, in_channels: int, input_size: tuple[int, int],
                     pretrained: str | None = None, width: int = 32, dropout: float = 0.25,
                     **_: Any) -> Any:
    """2 × (3×3 conv → batch norm → ReLU → 2×2 max-pool), dropout, then a linear layer."""
    from torch import nn

    _no_pretrained("SimpleCNN", pretrained)
    if min(input_size) < 8:
        raise ValueError(f"SimpleCNN needs inputs of at least 8×8, got {input_size[0]}×{input_size[1]}")

    def block(inputs: int, outputs: int) -> list[nn.Module]:
        return [nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
                nn.MaxPool2d(2)]

    features = nn.Sequential(*block(in_channels, width), *block(width, width * 2), nn.Flatten())
    flat = _flat_features(features, in_channels, input_size)
    return nn.Sequential(features, nn.Dropout(dropout), nn.Linear(flat, num_classes))


def build_lenet5(*, num_classes: int, in_channels: int, input_size: tuple[int, int],
                 pretrained: str | None = None, **_: Any) -> Any:
    """LeNet-5 (LeCun et al., 1998) in its common modern form: ReLU and max-pooling instead of
    the scaled tanh and subsampling. Designed for 32×32 inputs."""
    from torch import nn

    _no_pretrained("LeNet-5", pretrained)
    if min(input_size) < 16:
        raise ValueError(f"LeNet-5 needs inputs of at least 16×16, got {input_size[0]}×{input_size[1]}")
    features = nn.Sequential(nn.Conv2d(in_channels, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
    flat = _flat_features(features, in_channels, input_size)
    return nn.Sequential(features, nn.Linear(flat, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
                         nn.Linear(84, num_classes))


build_lenet5.input_size = (32, 32)  # type: ignore[attr-defined]  # what the runner resizes to by default


def build_resnet18(*, num_classes: int, in_channels: int, input_size: tuple[int, int],
                   pretrained: str | None = None, **_: Any) -> Any:
    """torchvision's ResNet-18 with a new classifier. Trained from scratch on small images
    (under 64 px) it uses a 3×3 stride-1 stem without the max-pool, as is usual for CIFAR;
    with ImageNet weights it keeps the original stem."""
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    if pretrained not in (None, "imagenet1k_v1"):
        raise ValueError(f"ResNet-18 has no weights called {pretrained!r}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if not pretrained and min(input_size) < 64:
        model.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        rgb = model.conv1.weight.data
        model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:  # one grey channel sees what the three colour channels saw together
            with torch.no_grad():
                model.conv1.weight.copy_(rgb.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                                         / in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
