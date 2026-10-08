"""MLflow: experiment tracking.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["MlflowPlugin"]


class MlflowPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("open_ui", "Open MLflow UI", "Start the MLflow UI on this workspace's store",
                   needs_connection=False),
    )
