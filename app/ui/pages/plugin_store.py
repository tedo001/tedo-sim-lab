"""Plugin Store: every integration with its live status, licence, requirements and account.
Install runs pip in the background (after accepting non-permissive terms, when a plugin has
them); credentials go to the OS keyring and are never shown again; sources can be browsed and
downloaded from (``plugin_browse.py``)."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QPushButton, QWidget

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, Pill, label
from .catalog_common import licence_pill, status_pill
from .plugin_browse import BrowseCard

__all__ = ["PluginStorePage"]

_SOURCE = {"env": ("Environment variable", "ok"), "keyring": ("OS keyring", "ok"),
           "missing": ("Not set", "planned")}
#: Plugins whose actions live on another page.
_PAGES = {"mlflow": ("mlflow", "Open the MLflow page"), "colab": ("colab", "Open the Google Colab page")}


def _licence(manifest) -> Pill:
    """The lab's own plugins say so; a third-party tool shows its licence category."""
    if manifest.origin != "builtin":
        return licence_pill(manifest.license)
    pill = Pill("Lab's own code", "planned")
    pill.setToolTip(manifest.license.name)
    return pill


class PluginStorePage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Plugin Store", "integrations: status, licence, install and account", parent)
        self.ctx = ctx
        self.names = [m.name for m in ctx.catalog.plugins.manifests()]
        self.selected: str | None = None
        self._test_job: str | None = None
        self._install_job: str | None = None

        listing = Card("Plugins", "pick one to see its details", padded=False)
        self.table = DataTable(("Plugin", "Does", "Status", "Licence"), stretch_column=1)
        self.table.itemSelectionChanged.connect(self._picked)
        listing.add(self.table)
        self.body.addWidget(listing)

        self.about = Card("About")
        self.facts = KeyValues([("Version", "—"), ("Author", "—"), ("Licence", "—"), ("Terms", "—"),
                                ("Needs", "—"), ("Home page", "—")])
        self.about.add(self.facts)
        self.description = label("", "Body", wrap=True)
        self.about.add(self.description)
        self.notes = label("", "CardCaption", wrap=True)
        self.about.add(self.notes)
        self.page_button = QPushButton(icon("external-link"), "")
        self.page_button.clicked.connect(self._open_page)
        self.about.add(self.page_button)
        self.actions_text = label("", "CardCaption", wrap=True)
        self.about.add(self.actions_text)
        self.body.addWidget(self.about)

        self.install_card = Card("Install", "pip, into the experiment Python")
        self.command = label("", "Mono", wrap=True)
        self.install_card.add(self.command)
        self.terms = QCheckBox("")
        self.terms.toggled.connect(lambda _: self._update_install())
        self.install_card.add(self.terms)
        self.install_button = QPushButton(icon("package-plus"), "Install")
        self.install_button.setObjectName("Primary")
        self.install_button.clicked.connect(self.install)
        self.install_card.add(self.install_button)
        self.install_text = label("", "Mono", wrap=True)
        self.install_card.add(self.install_text)
        self.body.addWidget(self.install_card)

        self.account = Card("Account", "stored in the OS keyring; values are never shown")
        self.account_table = DataTable(("Credential", "Variable", "Set in"), mono_columns=(1,),
                                       stretch_column=0)
        self.account.add(self.account_table)
        row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setReadOnly(True)
        self.key.setMaximumWidth(220)
        self.value = QLineEdit()
        self.value.setEchoMode(QLineEdit.EchoMode.Password)
        self.value.setPlaceholderText("Paste the value to store")
        self.value.textChanged.connect(lambda _: self._update_account())
        self.save_button = QPushButton(icon("key-round"), "Save")
        self.save_button.clicked.connect(self.save_credential)
        self.remove_button = QPushButton(icon("trash-2"), "Remove")
        self.remove_button.clicked.connect(self.remove_credential)
        for widget in (self.key, self.value, self.save_button, self.remove_button):
            row.addWidget(widget, 1 if widget is self.value else 0)
        self.account.add(row)
        self.account_table.itemSelectionChanged.connect(self._credential_picked)
        test_row = QHBoxLayout()
        self.test_button = QPushButton(icon("refresh-cw"), "Test connection")
        self.test_button.clicked.connect(self.test)
        test_row.addWidget(self.test_button)
        test_row.addStretch(1)
        self.account.add(test_row)
        self.account_text = label("", "Body", wrap=True)
        self.account.add(self.account_text)
        self.body.addWidget(self.account)

        self.browse = BrowseCard(ctx)
        self.body.addWidget(self.browse)
        self.body.addStretch(1)
        ctx.plugins.changed.connect(lambda _name: self.refresh())
        ctx.plugins.finished.connect(self._finished)
        self.refresh()
        if self.names:
            self.select(self.names[0])

    # Listing ------------------------------------------------------------------------------------
    def refresh(self) -> None:
        keep = self.selected
        self.table.blockSignals(True)
        self.table.clear_rows()
        for name in self.names:
            plugin = self.ctx.plugins.plugin(name)
            manifest = plugin.manifest
            self.table.add_row([manifest.title, manifest.description,
                                status_pill(plugin.status, manifest.planned_for), _licence(manifest)])
        self.table.blockSignals(False)
        if keep:
            self.select(keep)

    def select(self, name: str) -> None:
        self.table.blockSignals(True)
        self.table.selectRow(self.names.index(name))
        self.table.blockSignals(False)
        self._show(name)

    def _picked(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._show(self.names[rows[0].row()])

    # One plugin ---------------------------------------------------------------------------------
    def _show(self, name: str) -> None:
        changed = name != self.selected
        self.selected = name
        plugin = self.ctx.plugins.plugin(name)
        manifest, licence = plugin.manifest, plugin.manifest.license
        self.about.set_title(manifest.title)
        self.facts.set_value("Version", manifest.version)
        self.facts.set_value("Author", manifest.author)
        self.facts.set_value("Licence", licence.name)
        self.facts.set_value("Terms", licence.url or "—")
        self.facts.set_value("Needs", ", ".join(manifest.requires) or "nothing to install")
        self.facts.set_value("Home page", manifest.homepage or "—")
        self.description.setText(manifest.description)
        self.notes.setText(licence.notes or "")
        target = _PAGES.get(name)
        self.page_button.setVisible(target is not None and plugin.status not in ("planned",))
        self.page_button.setText(target[1] if target else "")
        lines = [f"• {a.label}: {a.description}" + ("" if a.enabled else f" — {a.disabled_reason}")
                 for a in plugin.actions()]
        self.actions_text.setText("What it does:\n" + "\n".join(lines) if lines else "")

        plan = plugin.install_plan()
        self.install_card.setVisible(bool(manifest.requires))
        self.command.setText(" ".join(plan.command) if plan.needed else "Everything it needs is installed.")
        self.terms.setVisible(plan.needed and plan.acknowledgement is not None)
        self.terms.setText(plan.acknowledgement or "")
        if changed:
            self.terms.setChecked(False)
            self.install_text.setText("")
            self.account_text.setText("")
            self.value.clear()
        self._update_install()

        self.account.setVisible(bool(manifest.credentials))
        self.account_table.clear_rows()
        for spec in manifest.credentials:
            text, tone = _SOURCE[self.ctx.credentials.source(spec.key)]
            title = spec.label + ("" if spec.required else " (optional)")
            self.account_table.add_row([title, spec.key, Pill(text, tone)])
        if manifest.credentials:
            self.account_table.selectRow(0)
        self._credential_picked()

        is_source = bool({"dataset_source", "model_source"} & manifest.capabilities)
        browsable = is_source and plugin.status in ("available", "not_connected")
        self.browse.setVisible(browsable)
        if browsable and (changed or self.browse.name != name):
            self.browse.set_plugin(name)
        self.browse.setEnabled(plugin.status == "available")
        self.browse.setToolTip("" if plugin.status == "available" else "Connect the account first")

    def _update_install(self) -> None:
        plugin = self.ctx.plugins.plugin(self.selected) if self.selected else None
        plan = plugin.install_plan() if plugin else None
        ready = bool(plan and plan.needed and plugin.manifest.maturity != "planned")
        accepted = plan is None or plan.acknowledgement is None or self.terms.isChecked()
        self.install_button.setEnabled(ready and accepted and self._install_job is None)
        self.install_button.setToolTip("" if ready else "Nothing to install" if plan and not plan.needed
                                       else "This plugin is planned; there is nothing to install yet")

    def _credential_picked(self) -> None:
        rows = self.account_table.selectionModel().selectedRows()
        manifest = self.ctx.plugins.plugin(self.selected).manifest if self.selected else None
        spec = manifest.credentials[rows[0].row()] if manifest and rows else None
        self.key.setText(spec.key if spec else "")
        self.value.setEchoMode(QLineEdit.EchoMode.Password if spec is None or spec.secret
                               else QLineEdit.EchoMode.Normal)
        self._update_account()

    def _update_account(self) -> None:
        key = self.key.text()
        keyring = self.ctx.credentials.keyring_available
        source = self.ctx.credentials.source(key) if key else "missing"
        self.save_button.setEnabled(bool(key and self.value.text().strip() and keyring))
        self.save_button.setToolTip("" if keyring else "No OS keyring here: set the environment variable")
        self.remove_button.setEnabled(source == "keyring")
        self.remove_button.setToolTip("" if source != "env" else "Set by an environment variable; "
                                      "unset it outside the lab")
        plugin = self.ctx.plugins.plugin(self.selected) if self.selected else None
        self.test_button.setEnabled(plugin is not None and self._test_job is None)

    # Acting ---------------------------------------------------------------------------------------
    def save_credential(self) -> None:
        problem = self.ctx.plugins.set_credential(self.key.text(), self.value.text())
        self.value.clear()
        self.account_text.setText(problem or f"{self.key.text()} stored in the OS keyring.")

    def remove_credential(self) -> None:
        problem = self.ctx.plugins.remove_credential(self.key.text())
        self.account_text.setText(problem or f"{self.key.text()} removed from the OS keyring.")

    def test(self) -> str | None:
        if not self.selected or self._test_job:
            return None
        self.account_text.setText("Testing…")
        self._test_job = self.ctx.plugins.test(self.selected)
        self._update_account()
        return self._test_job

    def install(self) -> str | None:
        if not self.install_button.isEnabled():
            return None
        self.install_text.setText("Installing… (pip output appears here if it fails)")
        self._install_job = self.ctx.plugins.install(self.selected)
        self._update_install()
        return self._install_job

    def _finished(self, job_id: str, name: str, action: str, result: object, error: str) -> None:
        if job_id == self._test_job:
            self._test_job = None
            message = error or getattr(result, "message", "")
            ok = not error and getattr(result, "ok", False)
            self.account_text.setText(("Connected. " if ok else "Not connected. ") + message)
            self._update_account()
        elif job_id == self._install_job:
            self._install_job = None
            self.install_text.setText(f"Install failed:\n{error}" if error else "Installed.")
            self.refresh()

    def _open_page(self) -> None:
        target = _PAGES.get(self.selected or "")
        if target:
            self.ctx.navigate(target[0])
