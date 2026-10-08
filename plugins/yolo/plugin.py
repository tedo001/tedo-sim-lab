"""Ultralytics YOLO: yOLO detection, segmentation and pose models through the Ultralytics package.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["YoloPlugin"]


class YoloPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("train", "Train", "Train a YOLO model on a detection dataset",
                   needs_connection=False),
        ActionSpec("predict", "Predict", "Run a YOLO model on images",
                   needs_connection=False),
    )
