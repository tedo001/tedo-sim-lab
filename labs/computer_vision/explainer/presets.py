"""The networks the explainer offers: TinyVGG sized for each of the lab's image
datasets. Class names come from the dataset cards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .network import Architecture, tiny_vgg
from .samples import Sample, digit_samples, pattern_samples

__all__ = ["PRESETS", "Preset", "preset"]


@dataclass(frozen=True)
class Preset:
    id: str
    dataset_id: str
    dataset_name: str
    channels: int
    size: int
    #: Digit samples suit MNIST-style networks.
    digits: bool = False

    @property
    def title(self) -> str:
        colour = "grey" if self.channels == 1 else "colour"
        return f"TinyVGG · {self.dataset_name} ({self.size}×{self.size} {colour})"

    def architecture(self, class_names: Sequence[str] = ()) -> Architecture:
        names = tuple(class_names) or tuple(f"class {i}" for i in range(10))
        return tiny_vgg(in_channels=self.channels, size=self.size, class_names=names,
                        name=f"TinyVGG for {self.dataset_name}")

    def samples(self) -> list[Sample]:
        found = digit_samples(self.size, self.channels) if self.digits else []
        return found + pattern_samples(self.size, self.channels)


PRESETS: tuple[Preset, ...] = (
    Preset("tiny_vgg-mnist", "mnist", "MNIST", 1, 28, digits=True),
    Preset("tiny_vgg-fashion_mnist", "fashion_mnist", "Fashion-MNIST", 1, 28),
    Preset("tiny_vgg-cifar10", "cifar10", "CIFAR-10", 3, 32),
)


def preset(preset_id: str) -> Preset:
    for candidate in PRESETS:
        if candidate.id == preset_id:
            return candidate
    raise KeyError(preset_id)
