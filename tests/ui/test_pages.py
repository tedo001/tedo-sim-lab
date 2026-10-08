"""The pages built in phase 1: Home, Documentation, Settings."""

from __future__ import annotations

from app.navigation import NAV
from app.ui.pages.documentation import DocumentationPage, find_documents
from app.ui.pages.home import HomePage
from app.ui.pages.settings import SettingsPage
from app.ui.widgets import Pill
from core.common import CredentialStore
from core.common.secrets import KEYRING_SERVICE


def test_home_lists_every_page(ctx, qtbot) -> None:
    home = HomePage(ctx)
    qtbot.addWidget(home)
    assert home.status_table.rowCount() == len(NAV)
    assert home.workspace_values.value_text("Workspace") == str(ctx.paths.workspace)


def test_home_buttons_navigate(ctx, qtbot) -> None:
    from PyQt6.QtWidgets import QPushButton
    visited = []
    ctx.navigate = visited.append
    home = HomePage(ctx)
    qtbot.addWidget(home)
    for button in home.findChildren(QPushButton):
        if button.text() in ("Documentation", "Settings"):
            button.click()
    assert visited == ["documentation", "settings"]


def test_settings_shows_sources_never_values(ctx, keyring_backend, qtbot) -> None:
    from dataclasses import replace
    keyring_backend.set_password(KEYRING_SERVICE, "KAGGLE_KEY", "kaggle-hidden-123")
    store = CredentialStore(env={"HF_TOKEN": "hf_hiddenvalue000000000000000"}, backend=keyring_backend)
    settings = SettingsPage(replace(ctx, credentials=store))
    qtbot.addWidget(settings)
    table = settings.credential_table
    sources = {table.item(row, 1).text(): table.cellWidget(row, 2).findChild(Pill).text()
               for row in range(table.rowCount())}
    assert sources["HF_TOKEN"] == "Environment variable"
    assert sources["KAGGLE_KEY"] == "OS keyring"
    assert sources["GITHUB_TOKEN"] == "Not set"
    from PyQt6.QtWidgets import QLabel
    shown = " ".join(widget.text() for widget in settings.findChildren(QLabel))
    assert "hidden" not in shown


def test_documentation_lists_repo_docs(ctx, qtbot) -> None:
    documents = find_documents(ctx.paths.code_root)
    assert any(doc.name == "CLAUDE.md" for doc in documents)
    page = DocumentationPage(ctx)
    qtbot.addWidget(page)
    assert page.index.count() == len(documents)
    assert page.view.toPlainText().strip()


def test_documentation_empty_state(ctx, qtbot, tmp_path) -> None:
    from dataclasses import replace
    empty = replace(ctx, paths=replace(ctx.paths, code_root=tmp_path))
    page = DocumentationPage(empty)
    qtbot.addWidget(page)
    assert page.index.count() == 0
