"""Trained TinyVGG networks from finished runs: what the run folder says about them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labs.computer_vision.explainer import PRESETS, Preset
from labs.computer_vision.export import EXPLAINER_META, EXPLAINER_WEIGHTS

__all__ = ["EXPLAINER_WEIGHTS", "preset_for", "trained_meta"]


def trained_meta(run_dir: Path) -> dict[str, Any]:
    """``explainer.json`` of a run: dataset, input shape, class names, metrics."""
    return json.loads((run_dir / EXPLAINER_META).read_text(encoding="utf-8"))


def preset_for(meta: dict[str, Any]) -> Preset:
    """The preset with the same data and input shape (for samples and drawing), or a new one."""
    for candidate in PRESETS:
        if (candidate.dataset_id, candidate.channels, candidate.size) == (
                meta["dataset"], meta["in_channels"], meta["size"]):
            return candidate
    return Preset(f"run-{meta['dataset']}", meta["dataset"], meta["dataset"], meta["in_channels"],
                  meta["size"], digits=meta["dataset"] == "mnist")
