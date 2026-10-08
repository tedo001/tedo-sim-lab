"""Hand a trained TinyVGG to the CNN Explainer.

The explainer feeds pixel values 0..1 straight into the network. When the model was
trained on normalised inputs ((x − mean) / std), that step is folded into the first
convolution's weights and bias, which is exact because TinyVGG's first convolution has
no padding. The result is ``explainer.npz`` (weights) and ``explainer.json`` (what they
are for) in the run folder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .explainer.network import ExplainerNet, tiny_vgg

__all__ = ["EXPLAINER_META", "EXPLAINER_WEIGHTS", "export_for_explainer", "fold_normalization"]

EXPLAINER_WEIGHTS = "explainer.npz"
EXPLAINER_META = "explainer.json"


def fold_normalization(params: Mapping[str, np.ndarray], layer: str, mean: Sequence[float],
                       std: Sequence[float]) -> dict[str, np.ndarray]:
    """Weights that give on raw inputs what ``params`` give on (x − mean) / std."""
    folded = dict(params)
    weight = np.asarray(params[f"{layer}.weight"], dtype=np.float64)
    scale = 1.0 / np.asarray(std, dtype=np.float64)
    shift = np.asarray(mean, dtype=np.float64) * scale
    folded[f"{layer}.weight"] = weight * scale[None, :, None, None]
    folded[f"{layer}.bias"] = np.asarray(params[f"{layer}.bias"]) - np.einsum("ocij,c->o", weight, shift)
    return folded


def export_for_explainer(state_dict: Mapping[str, Any], run_dir: Path, *, dataset: str,
                         in_channels: int, size: int, class_names: Sequence[str], hidden: int = 10,
                         normalization: tuple[Sequence[float], Sequence[float]] | None = None,
                         metrics: Mapping[str, float] | None = None) -> Path:
    params = {name: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
              for name, value in state_dict.items()}
    architecture = tiny_vgg(in_channels=in_channels, size=size, class_names=tuple(class_names), hidden=hidden)
    first = architecture.layers[0]
    if normalization is not None:
        if first.kind != "conv" or first.padding:
            raise ValueError("normalisation can only be folded into an unpadded first convolution")
        params = fold_normalization(params, first.name, *normalization)
    net = ExplainerNet(architecture, params, trained=True, source=run_dir.name)
    net.save_npz(run_dir / EXPLAINER_WEIGHTS)
    meta = {"dataset": dataset, "in_channels": in_channels, "size": size, "hidden": hidden,
            "class_names": list(class_names), "normalization_folded": normalization is not None,
            "metrics": dict(metrics or {})}
    (run_dir / EXPLAINER_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir / EXPLAINER_WEIGHTS
