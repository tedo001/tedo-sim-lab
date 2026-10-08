"""Settings: where things are, what is configured, and which credentials are set.

Read-only in build phase 1. Credential values are never displayed — only
where each one comes from. Storing credentials from the app arrives with the
Plugin Store (build phase 9); until then, set environment variables.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from core.common import KNOWN_CREDENTIALS, experiment_python, mlflow_tracking_uri

from ...services.context import AppContext
from ..widgets import Card, DataTable, KeyValues, Page, Pill, label

__all__ = ["SettingsPage"]

_SOURCE = {"env": ("Environment variable", "ok"), "keyring": ("OS keyring", "ok"),
           "missing": ("Not set", "planned")}


class SettingsPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Settings", str(ctx.paths.settings_file), parent)
        paths, config = ctx.paths, ctx.config

        workspace = Card("Workspace", "runtime data; git-ignored")
        workspace.add(KeyValues((
            ("Workspace", paths.workspace),
            ("Datasets", paths.datasets),
            ("Models", paths.models),
            ("Experiments", paths.experiments),
            ("Database", paths.database),
            ("MLflow artifacts", paths.mlruns),
            ("Logs", paths.logs),
        )))
        self.body.addWidget(workspace)

        configuration = Card("Configuration", "settings.yaml; defaults where unset")
        configuration.add(KeyValues((
            ("Log level", config.log_level),
            ("MLflow tracking URI", mlflow_tracking_uri(config, paths)),
            ("Terminal shell", config.terminal_shell),
            ("Concurrent runs", str(config.max_concurrent_runs)),
            ("Default device", config.default_device),
            ("Experiment Python", config.python_executable
             or f"{experiment_python(config)} (this app's Python)"),
        )))
        configuration.add(label("Edit settings.yaml and restart to change these. Editing from "
                                "this page is not built yet.", "CardCaption", wrap=True))
        self.body.addWidget(configuration)

        credentials = Card("Credentials", "values are never shown", padded=False)
        self.credential_table = DataTable(("Credential", "Variable", "Source"), mono_columns=(1,),
                                          stretch_column=2)
        for key, title in KNOWN_CREDENTIALS.items():
            text, tone = _SOURCE[ctx.credentials.source(key)]
            self.credential_table.add_row((title, key, Pill(text, tone)))
        credentials.add(self.credential_table)
        keyring = ("The OS keyring is available." if ctx.credentials.keyring_available
                   else "No OS keyring was found; use environment variables.")
        note = label(f"{keyring} Connecting accounts from the Plugin Store arrives in build "
                     "phase 9.", "CardCaption", wrap=True)
        note.setContentsMargins(14, 8, 14, 12)
        credentials.add(note)
        self.body.addWidget(credentials)
        self.body.addStretch(1)
