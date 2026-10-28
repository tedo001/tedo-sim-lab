"""A few examples of a downloaded dataset, for the Dataset Hub's preview: texts with their labels,
spectrogram or MNIST-style images with their labels, or the first rows of a table. Read without
PyTorch, so the window can show them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_registry.adapter import DatasetAdapter, TabularAdapter, TextAdapter

from .explore import IDX_SETS, SPECTROGRAM_SETS, _file, read_idx

__all__ = ["previewable", "samples"]

TEXT_CHARS = 400


def previewable(adapter: DatasetAdapter) -> bool:
    name = type(adapter).__name__
    return isinstance(adapter, (TabularAdapter, TextAdapter)) or name in IDX_SETS or name in SPECTROGRAM_SETS


def _pick(count: int, limit: int, seed: int) -> list[int]:
    return sorted(np.random.default_rng(seed).permutation(count)[:limit].tolist())


def samples(adapter: DatasetAdapter, root: Path, *, limit: int = 8, seed: int = 0) -> dict[str, Any]:
    """``{"kind": "text", "rows": [[label, text], ...]}``, ``{"kind": "images", "images": [[label,
    uint8 array], ...]}`` or ``{"kind": "table", "columns": [...], "rows": [[...], ...]}``."""
    name = type(adapter).__name__
    if isinstance(adapter, TextAdapter):
        texts, labels = adapter.load_texts(root, "test" if adapter.has_test_split else "all")
        classes = adapter.classes
        rows = [[classes[labels[i]], " ".join(texts[i].split())[:TEXT_CHARS]]
                for i in _pick(len(texts), limit, seed)]
        return {"kind": "text", "rows": rows, "total": len(texts)}
    if isinstance(adapter, TabularAdapter):
        features, target = adapter.load_frame(root)
        frame = features if target is None else features.assign(**{str(target.name): target})
        head = frame.head(limit)
        return {"kind": "table", "columns": [str(c) for c in head.columns],
                "rows": head.astype(str).values.tolist(), "total": len(frame)}
    if name in SPECTROGRAM_SETS:
        found = adapter.features(root)  # type: ignore[attr-defined]
        images, labels = found["images"], found["digits"]
        caption = "log mel spectrogram, time → right, pitch ↑"
    elif name in IDX_SETS:
        folder = adapter.data_dir(root) / IDX_SETS[name]
        images = read_idx(_file(folder, "t10k-images-idx3-ubyte"))
        labels = read_idx(_file(folder, "t10k-labels-idx1-ubyte"))
        caption = "test images"
    else:
        raise ValueError(f"{adapter.card.name} has no preview")
    classes: Any = getattr(adapter, "classes", ())
    chosen = _pick(len(images), limit, seed)
    return {"kind": "images", "caption": caption, "total": len(images),
            "images": [[str(classes[int(labels[i])]) if len(classes) else str(int(labels[i])), images[i]]
                       for i in chosen]}
