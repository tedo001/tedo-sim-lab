"""Google Colab: export an experiment as a Colab notebook and import its results back.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["ColabPlugin"]


class ColabPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("export_notebook", "Export notebook", "Write a training notebook for an experiment",
                   needs_connection=False),
        ActionSpec("import_results", "Import results", "Bring a Colab run's results into the lab",
                   needs_connection=False),
    )
