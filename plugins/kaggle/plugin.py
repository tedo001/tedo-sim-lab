"""Kaggle: search and download Kaggle datasets with your Kaggle API token.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["KagglePlugin"]


class KagglePlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("search_datasets", "Search datasets", "Find datasets on Kaggle by keyword",
                   needs_connection=True),
        ActionSpec("download_dataset", "Download dataset", "Download a dataset into the workspace",
                   needs_connection=True),
    )
