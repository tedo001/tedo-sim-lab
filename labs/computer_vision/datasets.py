"""Image-classification datasets: MNIST, Fashion-MNIST and CIFAR-10.

Files go to ``<workspace>/datasets/<card id>/`` in torchvision's layout, so torchvision
reads them. Downloads only happen when the person asks (the adapter contract checks the
licence first); the mirrors, file names and MD5 sums are torchvision's own. Checking what
is on disk needs no PyTorch, so the interface can ask quickly.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from core.common.cancel import CancelToken, ProgressFn
from core.dataset_registry.adapter import DatasetStats, ImageClassificationAdapter, LocalState

from ..common.download import download_file

__all__ = ["Cifar10Adapter", "FashionMnistAdapter", "MnistAdapter", "VisionAdapter", "targets_of"]


def targets_of(dataset: Any) -> np.ndarray:
    """Class index of every sample of a torchvision dataset (or a ``Subset`` of one)."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return targets_of(dataset.dataset)[np.asarray(dataset.indices)]
    return np.asarray(dataset.targets, dtype=np.int64)


def _scaled(progress: ProgressFn, part: int, parts: int) -> ProgressFn:
    def report(fraction: float, message: str) -> None:
        progress((part + max(fraction, 0.0)) / parts if fraction >= 0 else -1.0, message)
    return report


class VisionAdapter(ImageClassificationAdapter):
    """A torchvision image dataset with known files."""

    #: torchvision class name; also the folder torchvision expects under the dataset root.
    torchvision_name: ClassVar[str]
    #: Per-channel mean and standard deviation of the training images (values 0..1).
    mean: ClassVar[tuple[float, ...]]
    std: ClassVar[tuple[float, ...]]
    #: Files that must exist (relative to the data folder) for the dataset to be usable.
    required: ClassVar[tuple[str, ...]]

    @property
    def classes(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(self.card.class_names) or tuple(str(i) for i in range(self.card.num_classes or 0))

    def local_status(self, root: Path) -> LocalState:
        folder = self.data_dir(root)
        found = sum((folder / name).is_file() for name in self.required)
        if found == len(self.required):
            return "present"
        started = found or (folder.is_dir() and any(folder.rglob("*.gz")))
        return "partial" if started else "absent"

    def describe(self, root: Path) -> DatasetStats:
        return DatasetStats(dict(self.card.splits), self.card.num_classes, self.classes,
                            {"in_channels": self.in_channels, "native_size": self.native_size,
                             "mean": self.mean, "std": self.std})

    def load(self, root: Path, split: Literal["train", "test"],
             transform: Callable[[Any], Any] | None = None) -> Any:
        if self.local_status(root) != "present":
            raise FileNotFoundError(f"{self.card.name} is not downloaded. Download it from the "
                                    "Experiment Builder or the Computer Vision lab first.")
        from torchvision import datasets

        dataset_class = getattr(datasets, self.torchvision_name)
        return dataset_class(str(self.data_dir(root)), train=split == "train", transform=transform,
                             download=False)


class _MnistLike(VisionAdapter):
    in_channels = 1
    native_size = (28, 28)
    mirrors: ClassVar[tuple[str, ...]]
    resources: ClassVar[tuple[tuple[str, str], ...]]

    @classmethod
    def _raw(cls) -> str:
        return f"{cls.torchvision_name}/raw"

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        raw = self.data_dir(root) / self._raw()
        for part, (filename, md5) in enumerate(self.resources):
            archive = download_file([mirror + filename for mirror in self.mirrors], raw / filename,
                                    md5=md5, progress=_scaled(progress, part, len(self.resources)),
                                    cancel=cancel)
            target = raw / filename.removesuffix(".gz")
            with gzip.open(archive, "rb") as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
        progress(1.0, f"{self.card.name} is ready")


_MNIST_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte")


class MnistAdapter(_MnistLike):
    torchvision_name = "MNIST"
    mean, std = (0.1307,), (0.3081,)
    mirrors = ("https://ossci-datasets.s3.amazonaws.com/mnist/", "http://yann.lecun.com/exdb/mnist/")
    resources = (("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                 ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
                 ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
                 ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"))
    required = tuple(f"MNIST/raw/{name}" for name in _MNIST_FILES)


class FashionMnistAdapter(_MnistLike):
    torchvision_name = "FashionMNIST"
    mean, std = (0.2860,), (0.3530,)
    mirrors = ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",)
    resources = (("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
                 ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
                 ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
                 ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"))
    required = tuple(f"FashionMNIST/raw/{name}" for name in _MNIST_FILES)


class Cifar10Adapter(VisionAdapter):
    torchvision_name = "CIFAR10"
    in_channels = 3
    native_size = (32, 32)
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    md5 = "c58f30108f718f92721af3b95e74349a"
    required = tuple(f"cifar-10-batches-py/{name}" for name in
                     ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",
                      "test_batch", "batches.meta"))

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        folder = self.data_dir(root)
        archive = download_file([self.url], folder / "cifar-10-python.tar.gz", md5=self.md5,
                                progress=_scaled(progress, 0, 1), cancel=cancel)
        progress(-1.0, "Unpacking CIFAR-10…")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(folder, filter="data")
        progress(1.0, f"{self.card.name} is ready")
