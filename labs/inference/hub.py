"""Where downloaded models live and which of a Hub repository's files a model needs.

Pretrained models are downloaded through the Hugging Face plugin (licence shown first, gated and
private repositories refused) into ``<workspace>/models/huggingface/<owner>__<name>/``. A
repository often holds the same weights in several formats; the lab fetches only what PyTorch
and ``transformers`` read: configuration and tokenizer files and one set of weights, preferring
``safetensors``.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from core.model_registry.cards import HUB_FOLDER, hub_folder

__all__ = ["HUB_FOLDER", "is_downloaded", "model_folder", "needed_files"]

_KEEP = ("*.json", "*.txt", "*.model", "tokenizer*", "*.tiktoken", "merges.txt", "vocab.*", "*.py.json")
_SKIP_DIRS = ("onnx/", "coreml/", "openvino/", "tflite/", "flax/", "runs/", "original/")
_SKIP = ("tf_model.h5", "flax_model.msgpack", "rust_model.ot", "*.onnx", "*.onnx_data", "*.gguf", "*.ckpt",
         "*.mlmodel", "*.tflite", "*.msgpack", "*.h5", "*.pkl", "*.pickle")


def model_folder(workspace: Path, repo: str) -> Path:
    return hub_folder(workspace, repo)


def is_downloaded(workspace: Path, repo: str) -> bool:
    folder = model_folder(workspace, repo)
    return (folder / "config.json").is_file() or (folder / "preprocessor_config.json").is_file()


def needed_files(files: list[str]) -> list[str]:
    """The files of a model repository worth downloading (see the module notes). Weights: every
    ``*.safetensors`` shard if there are any, else the PyTorch ``*.bin`` files; never pickled
    formats other than PyTorch's own."""
    def skipped(name: str) -> bool:
        low = name.lower()
        return any(low.startswith(d) or f"/{d}" in low for d in _SKIP_DIRS) or \
            any(fnmatch.fnmatch(Path(low).name, pattern) for pattern in _SKIP)

    files = [name for name in files if not skipped(name)]
    safetensors = [name for name in files if name.endswith(".safetensors")]
    weights = safetensors or [name for name in files if name.endswith(".bin") and "training_args" not in name]
    keep = [name for name in files if any(fnmatch.fnmatch(Path(name).name, p) for p in _KEEP)]
    return sorted(set(keep) | set(weights))
