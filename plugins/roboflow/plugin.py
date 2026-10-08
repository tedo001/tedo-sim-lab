"""Roboflow: import datasets from Roboflow Universe and your Roboflow projects.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["RoboflowPlugin"]


class RoboflowPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("list_projects", "List projects", "Your Roboflow workspaces and projects",
                   needs_connection=True),
        ActionSpec("download_dataset", "Download dataset", "Export a dataset version into the workspace",
                   needs_connection=True),
    )
