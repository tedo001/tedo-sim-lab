"""Google Colab: export an experiment as a Colab notebook and read back the results zip its last
cell downloads (see :mod:`plugins.colab.notebook`). The Google Colab page drives both."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.plugin_api import ActionSpec, ManifestPlugin

from .notebook import build_notebook, read_results

__all__ = ["ColabPlugin"]


class ColabPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("export_notebook", "Export notebook", "Write a training notebook for an experiment"),
        ActionSpec("import_results", "Import results", "Read a Colab run's results zip"),
    )

    def _run(self, action: str, **kwargs: Any) -> Any:
        if action == "export_notebook":
            return build_notebook(kwargs["spec"], dataset=kwargs["dataset"], repository=kwargs["repository"],
                                  revision=kwargs["revision"], run_id=kwargs["run_id"])
        if action == "import_results":
            return read_results(Path(kwargs["archive"]))
        return super()._run(action, **kwargs)
