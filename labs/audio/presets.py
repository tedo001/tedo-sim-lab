"""Preset audio experiments: spoken digits as log mel spectrograms, trained with the lab's image
networks (the PyTorch runner), so the Model Testing page can show what each layer hears."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.common.vocab import Task
from core.experiment_engine.spec import ExperimentSpec

from ..computer_vision.presets import image_spec

__all__ = ["AUDIO_PRESETS", "AudioPreset"]


@dataclass(frozen=True)
class AudioPreset:
    id: str
    title: str
    summary: str
    dataset: str
    cost: str
    make: Callable[[], ExperimentSpec]


AUDIO_PRESETS: tuple[AudioPreset, ...] = (
    AudioPreset("digits_simple_cnn", "Spoken digits · SimpleCNN", "Ten spoken digits from six speakers; "
                "about 93% on held-out recordings.", "spoken_digits", "a minute on a CPU",
                lambda: image_spec("spoken-digits-simple-cnn", "spoken_digits", "simple_cnn", 10,
                                   augmentation=("random_crop",), task=Task.AUDIO_CLASSIFICATION)),
    AudioPreset("digits_tiny_vgg", "Spoken digits · TinyVGG", "The CNN Explainer's network on spectrograms; "
                "open the run on Model Testing to see each layer's maps and Grad-CAM.", "spoken_digits",
                "a minute on a CPU",
                lambda: image_spec("spoken-digits-tiny-vgg", "spoken_digits", "tiny_vgg", 10,
                                   augmentation=("random_crop",), task=Task.AUDIO_CLASSIFICATION)),
)
