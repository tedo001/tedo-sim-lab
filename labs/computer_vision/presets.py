"""Ready-made image-classification experiments for the Computer Vision lab. Each opens in
the Experiment Builder, where every setting can still be changed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.common.vocab import Task
from core.experiment_engine.spec import (
    CheckpointSpec,
    DataSpec,
    EarlyStoppingSpec,
    ExperimentSpec,
    ModelSpec,
    OptimizerSpec,
    SchedulerSpec,
    TorchTrainingSpec,
    TransformSpec,
)

__all__ = ["CV_PRESETS", "CvPreset", "image_spec"]

_NORMALIZE = (TransformSpec(name="normalize"),)


@dataclass(frozen=True)
class CvPreset:
    id: str
    title: str
    summary: str
    dataset: str
    #: Rough time on a laptop CPU / a 4 GB GPU, for the person choosing.
    cost: str
    make: Callable[[], ExperimentSpec]


def image_spec(name: str, dataset: str, model: str, epochs: int, *, augmentation: tuple[str, ...] = (),
               lr: float = 1e-3, scheduler: str = "none", patience: int | None = None,
               params: dict | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        name=name, task=Task.IMAGE_CLASSIFICATION,
        data=DataSpec(dataset=dataset, preprocessing=_NORMALIZE,
                      augmentation=tuple(TransformSpec(name=step) for step in augmentation)),
        model=ModelSpec(model=model, params=params or {}),
        training=TorchTrainingSpec(epochs=epochs, batch_size=64, optimizer=OptimizerSpec(name="adam", lr=lr),
                                   scheduler=SchedulerSpec(name=scheduler),
                                   checkpoint=CheckpointSpec(monitor="val_acc", mode="max"),
                                   early_stopping=EarlyStoppingSpec(monitor="val_acc", mode="max",
                                                                    patience=patience) if patience else None))


CV_PRESETS: tuple[CvPreset, ...] = (
    CvPreset("mnist_simple_cnn", "MNIST · SimpleCNN", "Handwritten digits with a small two-block CNN; "
             "reaches about 99% test accuracy.", "mnist", "a few minutes on a CPU",
             lambda: image_spec("mnist-simple-cnn", "mnist", "simple_cnn", 3)),
    CvPreset("mnist_tiny_vgg", "MNIST · TinyVGG", "The CNN Explainer's network. When it finishes, the "
             "CNN Explainer can show what it learned.", "mnist", "a few minutes on a CPU",
             lambda: image_spec("mnist-tiny-vgg", "mnist", "tiny_vgg", 5)),
    CvPreset("fashion_lenet5", "Fashion-MNIST · LeNet-5", "Clothing images with the classic LeNet-5, "
             "resized to 32×32.", "fashion_mnist", "about 10 minutes on a CPU",
             lambda: image_spec("fashion-mnist-lenet5", "fashion_mnist", "lenet5", 10, patience=3)),
    CvPreset("cifar10_resnet18", "CIFAR-10 · ResNet-18", "Colour photos of 10 object classes with a "
             "residual network, crop-and-flip augmentation and a cosine schedule.", "cifar10",
             "a GPU is recommended (a few minutes per epoch on 4 GB)",
             lambda: image_spec("cifar10-resnet18", "cifar10", "resnet18", 20,
                           augmentation=("random_crop", "horizontal_flip"), scheduler="cosine", patience=5)),
)
