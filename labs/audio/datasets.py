"""Spoken digits as images: the Free Spoken Digit Dataset (Jackson et al., CC BY-SA 4.0) — 3,000
recordings of the digits 0–9, six speakers, fifty of each digit per speaker, 8 kHz WAV.

Each clip becomes its 64×64 log mel spectrogram (:func:`labs.audio.dsp.log_mel_image`), so the
lab's image networks (SimpleCNN, LeNet-5, TinyVGG, ResNet-18, layer stacks) train on it unchanged
and the Model Testing page can show what every layer makes of a sound. The test split is the
dataset's own: recordings 0–4 of every digit and speaker.

Files come one by one from the dataset's GitHub repository at a fixed tag; the spectrograms are
worked out once and kept in ``features.npz`` beside the recordings."""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from core.common.cancel import Cancelled, CancelToken, ProgressFn

from ..common.download import download_file
from ..computer_vision.datasets import VisionAdapter

__all__ = ["SPEAKERS", "SpokenDigitsAdapter", "SpokenDigitsSet", "file_names"]

SPEAKERS = ("george", "jackson", "lucas", "nicolas", "theo", "yweweler")
TAG = "v1.0.10"
BASE = f"https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/{TAG}/recordings/"
_NAME = re.compile(r"^(\d)_([a-z]+)_(\d+)\.wav$")
TEST_TAKES = range(5)


def file_names() -> list[str]:
    return [f"{digit}_{speaker}_{take}.wav" for speaker in SPEAKERS for digit in range(10)
            for take in range(50)]


class SpokenDigitsSet:
    """A torch-style dataset of spectrogram images (PIL, mode "L") with ``targets``."""

    def __init__(self, images: np.ndarray, targets: np.ndarray, transform: Callable[[Any],
                                                                                    Any] | None) -> None:
        self.images, self.targets, self.transform = images, targets.astype(np.int64), transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        image = Image.fromarray(self.images[index], mode="L")
        return (self.transform(image) if self.transform else image), int(self.targets[index])


class SpokenDigitsAdapter(VisionAdapter):
    torchvision_name = "SpokenDigits"
    in_channels = 1
    native_size = (64, 64)
    mean, std = (0.215,), (0.284,)
    required: ClassVar[tuple[str, ...]] = tuple(f"recordings/{name}" for name in file_names())

    def local_status(self, root: Path) -> Literal["present", "partial", "absent"]:
        folder = self.data_dir(root) / "recordings"
        if not folder.is_dir():
            return "absent"
        found = sum(1 for _ in folder.glob("*.wav"))
        return "present" if found >= len(self.required) else ("partial" if found else "absent")

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        folder = self.data_dir(root) / "recordings"
        names = [n for n in file_names() if not (folder / n).is_file()]
        total, done = len(names), [0]

        def fetch(name: str) -> None:
            if cancel.cancelled:
                raise Cancelled()
            download_file([BASE + name], folder / name, progress=lambda *_: None, cancel=cancel, timeout=30)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == total:
                progress(done[0] / max(total, 1), f"{done[0]} of {total} recordings")

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(fetch, name) for name in names]:
                future.result()
        progress(1.0, f"{self.card.name} is ready")

    def features(self, root: Path) -> dict[str, np.ndarray]:
        """images (N×64×64 uint8), digits, speaker index and take of every recording."""
        folder = self.data_dir(root)
        cache = folder / "features.npz"
        if cache.is_file():
            with np.load(cache) as saved:
                return {key: saved[key] for key in saved.files}
        from .dsp import load_wav, log_mel_image

        rows = []
        for path in sorted((folder / "recordings").glob("*.wav")):
            match = _NAME.match(path.name)
            if match and match.group(2) in SPEAKERS:
                rows.append((path, int(match.group(1)), SPEAKERS.index(match.group(2)), int(match.group(3))))
        images = np.stack([log_mel_image(*load_wav(path)) for path, *_ in rows])
        data = {"images": images, "digits": np.array([r[1] for r in rows]),
                "speakers": np.array([r[2] for r in rows]), "takes": np.array([r[3] for r in rows])}
        np.savez_compressed(cache, **data)
        return data

    def load(self, root: Path, split: Literal["train", "test"],
             transform: Callable[[Any], Any] | None = None) -> Any:
        if self.local_status(root) != "present":
            raise FileNotFoundError(f"{self.card.name} is not downloaded. Download it from the Audio lab or "
                                    "the Experiment Builder first.")
        data = self.features(root)
        test = np.isin(data["takes"], list(TEST_TAKES))
        keep = test if split == "test" else ~test
        return SpokenDigitsSet(data["images"][keep], data["digits"][keep], transform)
