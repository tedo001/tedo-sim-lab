"""Hugging Face: models and datasets from the Hugging Face Hub.

Gated repositories need your token and accepted terms.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["HuggingFacePlugin"]


class HuggingFacePlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("search_models", "Search models", "Find models on the Hub",
                   needs_connection=False),
        ActionSpec("download_model", "Download model", "Download a model snapshot into the workspace",
                   needs_connection=False),
        ActionSpec("download_dataset", "Download dataset", "Download a dataset into the workspace",
                   needs_connection=False),
    )
