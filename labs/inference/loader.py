"""Loading models in the inference worker: the device (CUDA when there is one, else the CPU), the
precision (half on CUDA for the large generative models, full elsewhere) and a small cache, so a
model loaded once answers every later request at once. Two models at most stay loaded; the one
used longest ago is dropped first, which keeps a 4 GB GPU from filling up."""

from __future__ import annotations

import gc
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Loaded", "device", "dtype_for", "load", "resolve_model"]

Loaded = Any
_CACHE: OrderedDict[tuple[str, str], Loaded] = OrderedDict()
_LIMIT = 2


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_for(half: bool) -> Any:
    import torch

    return torch.float16 if half and device() == "cuda" else torch.float32


def resolve_model(value: str, workspace: Path) -> Path:
    """A model given as a folder, or as a Hub repository id downloaded into the workspace."""
    from .hub import model_folder

    path = Path(value)
    if path.is_dir():
        return path
    folder = model_folder(workspace, value)
    if folder.is_dir():
        return folder
    raise FileNotFoundError(f"{value} is not downloaded: download it in the lab first (it goes to {folder})")


def load(kind: str, folder: Path, factory: Callable[[Path], Loaded]) -> Loaded:
    """``factory(folder)`` once per (kind, folder); evicts the least recently used beyond the limit."""
    key = (kind, str(Path(folder).resolve()))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    while len(_CACHE) >= _LIMIT:
        _CACHE.popitem(last=False)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is part of the experiment environment
            pass
    loaded = factory(Path(folder))
    _CACHE[key] = loaded
    return loaded
