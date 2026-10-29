"""Settings: where things are, every setting in settings.yaml (edited here, checked and saved
together), the experiment Python, which credentials are set, and the lab's version with a check
for a newer release.

Credential values are never displayed — only where each one comes from; they are stored or
removed in the Plugin Store (OS keyring) or set as environment variables.
"""

from __future__ import annotations

import platform

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QWidget

from core.common import KNOWN_CREDENTIALS, save_config

from ... import __version__
from ...services.context import AppContext
from ...services.updates import UpdateResult
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, Pill, label
from .experiment_python_card import ExperimentPythonCard
from .remote_targets_card import RemoteTargetsCard
from .settings_form import SettingsForm

__all__ = ["SettingsPage"]

_SOURCE = {"env": ("Environment variable", "ok"), "keyring": ("OS keyring", "ok"),
           "missing": ("Not set", "planned")}


class SettingsPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Settings", str(ctx.paths.settings_file), parent)
        self.ctx = ctx
        paths = ctx.paths
        edit = QPushButton(icon("file-text"), "Edit settings.yaml")
        edit.setToolTip("Open the file in the Code page (saved changes apply at the next start)")
        edit.clicked.connect(self.edit_file)
        self.head.add_action(edit)

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
        self.form = SettingsForm(ctx)
        self.body.addWidget(self.form)
        self.experiment_python = ExperimentPythonCard(ctx)
        self.body.addWidget(self.experiment_python)
        self.remote_targets = RemoteTargetsCard(ctx)
        self.body.addWidget(self.remote_targets)

        credentials = Card("Credentials", "values are never shown", padded=False)
        self.credential_table = DataTable(("Credential", "Variable", "Source"), mono_columns=(1,),
                                          stretch_column=0)
        self.refresh_credentials()
        ctx.plugins.changed.connect(lambda _name: self.refresh_credentials())
        credentials.add(self.credential_table)
        keyring = ("The OS keyring is available." if ctx.credentials.keyring_available
                   else "No OS keyring was found; use environment variables.")
        note = label(f"{keyring} Connect accounts in the Plugin Store.", "CardCaption", wrap=True)
        note.setContentsMargins(14, 8, 14, 12)
        credentials.add(note)
        self.body.addWidget(credentials)
        self.body.addWidget(self._about())
        self.body.addStretch(1)
        ctx.updates.finished.connect(self.show_update)
        if ctx.updates.result is not None:
            self.show_update(ctx.updates.result)

    def refresh_credentials(self) -> None:
        self.credential_table.clear_rows()
        for key, title in KNOWN_CREDENTIALS.items():
            text, tone = _SOURCE[self.ctx.credentials.source(key)]
            self.credential_table.add_row((title, key, Pill(text, tone)))

    def edit_file(self) -> None:
        if not self.ctx.paths.settings_file.is_file():
            save_config(self.ctx.config, self.ctx.paths)  # the current settings, to edit from
        self.ctx.open_file(self.ctx.paths.settings_file, 1)

    # About and updates -----------------------------------------------------------------------------
    def _about(self) -> Card:
        about = Card("About", "TEDO AI Research Lab")
        about.add(KeyValues((("Version", __version__), ("Python", platform.python_version()),
                             ("Qt for Python", pyside_version), ("System", platform.platform(terse=True)))))
        row = QHBoxLayout()
        self.check_button = QPushButton(icon("refresh-cw"), "Check for updates")
        self.check_button.clicked.connect(self.check_updates)
        self.release_button = QPushButton(icon("external-link"), "Open the release page")
        self.release_button.setVisible(False)
        self.release_button.clicked.connect(self._open_release)
        self.at_start = QCheckBox("Check when the app starts")
        self.at_start.setChecked(self.ctx.config.check_updates)
        self.at_start.setToolTip("Asks GitHub for the latest release once at start; nothing is downloaded")
        self.at_start.toggled.connect(lambda on: self.ctx.settings.update(check_updates=on))
        for widget in (self.check_button, self.release_button, self.at_start):
            row.addWidget(widget)
        row.addStretch(1)
        about.add(row)
        self.update_text = label("Asks GitHub for the latest release; nothing is downloaded.", "CardCaption",
                                 wrap=True)
        about.add(self.update_text)
        return about

    def check_updates(self) -> None:
        self.check_button.setEnabled(False)
        self.update_text.setText("Checking…")
        self.ctx.updates.check()

    def show_update(self, result: UpdateResult) -> None:
        self.check_button.setEnabled(True)
        self.update_text.setText(result.summary())
        self.release_button.setVisible(bool(result.newer and result.url))

    def _open_release(self) -> None:
        result = self.ctx.updates.result
        if result is not None and result.url:
            QDesktopServices.openUrl(QUrl(result.url))
