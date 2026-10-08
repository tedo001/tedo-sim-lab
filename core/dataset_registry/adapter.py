"""The contract every dataset loader implements (concrete loaders live in ``labs/``).

``prepare`` enforces the licence policy before any byte is downloaded, so no
loader can fetch something the card does not allow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from core.common.cancel import CancelToken, ProgressFn
from core.common.licensing import download_policy

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from torch.utils.data import Dataset

    from .cards import DatasetCard

__all__ = ["AcknowledgementRequired", "DatasetAdapter", "DatasetStats", "DownloadNotAllowed",
           "ImageClassificationAdapter", "LocalState", "TabularAdapter"]

LocalState = Literal["present", "partial", "absent"]


class DownloadNotAllowed(PermissionError):
    """The card's licence or access rules forbid the lab from downloading it."""


class AcknowledgementRequired(PermissionError):
    """The person has to accept the licence terms before this download."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


@dataclass(frozen=True)
class DatasetStats:
    samples: dict[str, int]
    num_classes: int | None = None
    classes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    def __init__(self, card: DatasetCard) -> None:
        self.card = card

    def data_dir(self, root: Path) -> Path:
        """Where this dataset lives inside the workspace's ``datasets/`` folder."""
        return root / self.card.id

    @abstractmethod
    def local_status(self, root: Path) -> LocalState: ...

    def prepare(self, root: Path, progress: ProgressFn, cancel: CancelToken, *,
                acknowledged: bool = False) -> None:
        """Download (if needed and allowed) and verify the data under ``root``."""
        if self.local_status(root) == "present":
            progress(1.0, f"{self.card.name} is already on disk")
            return
        decision = download_policy(self.card)
        if not decision.allowed:
            raise DownloadNotAllowed(decision.reason)
        if decision.acknowledgement and not acknowledged:
            raise AcknowledgementRequired(decision.acknowledgement)
        self._download(root, progress, cancel)

    @abstractmethod
    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None: ...

    @abstractmethod
    def describe(self, root: Path) -> DatasetStats: ...


class ImageClassificationAdapter(DatasetAdapter):
    """Images with one label each (MNIST, CIFAR, ...)."""

    classes: tuple[str, ...]
    in_channels: int
    native_size: tuple[int, int]

    @abstractmethod
    def load(self, root: Path, split: Literal["train", "test"],
             transform: Callable[[Any], Any] | None = None) -> Dataset: ...


class TabularAdapter(DatasetAdapter):
    """Rows of features with an optional target column."""

    @abstractmethod
    def load_frame(self, root: Path) -> tuple[pd.DataFrame, pd.Series | None]: ...
