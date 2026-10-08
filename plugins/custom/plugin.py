"""Custom plugin (template): a starting point for your own integration.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ManifestPlugin

__all__ = ["CustomPlugin"]


class CustomPlugin(ManifestPlugin):
    ACTIONS = ()
