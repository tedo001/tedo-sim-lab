"""RT-DETR: rT-DETR detection transformer through the Ultralytics package.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["RtDetrPlugin"]


class RtDetrPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("train", "Train", "Train an RT-DETR model on a detection dataset",
                   needs_connection=False),
        ActionSpec("predict", "Predict", "Run an RT-DETR model on images",
                   needs_connection=False),
    )
