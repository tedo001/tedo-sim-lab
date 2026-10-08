"""Home for build phase 1: the workspace and which pages are live.

The dashboard (resources, active run, queue, recent experiments) replaces the
build-status card in build phase 3.
"""

from __future__ import annotations

import platform

from PySide6 import __version__ as PYSIDE_VERSION
from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common import mlflow_tracking_uri

from ... import __version__
from ...navigation import NAV, SECTIONS
from ...services.context import AppContext
from ..widgets import Card, DataTable, KeyValues, Page, Pill, label
from .placeholder import status_text

__all__ = ["HomePage"]

_SECTION_NAMES = dict(SECTIONS, home="Home", system="System")


def _tone(planned_for: str | None) -> str:
    if planned_for is None:
        return "ok"
    return "info" if planned_for == "v0.1" else "planned"


class HomePage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        built = sum(spec.built for spec in NAV)
        super().__init__("Home", f"v{__version__} · {built} of {len(NAV)} pages live · "
                                 f"workspace {ctx.paths.workspace}", parent)
        self.ctx = ctx
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._status_card(), 3, Qt.AlignmentFlag.AlignTop)
        side = QVBoxLayout()
        side.setSpacing(12)
        side.addWidget(self._workspace_card())
        if ctx.catalog.errors:
            side.addWidget(self._problems_card(ctx.catalog.errors))
        side.addStretch(1)
        row.addLayout(side, 2)
        self.body.addLayout(row)
        self.body.addStretch(1)

    def _status_card(self) -> Card:
        card = Card("Build status", "what works in this build", padded=False)
        self.status_table = DataTable(("Page", "Section", "Status"), stretch_column=2)
        for spec in NAV:
            self.status_table.add_row((spec.title, _SECTION_NAMES[spec.section],
                                       Pill(status_text(spec), _tone(spec.planned_for))))
        card.add(self.status_table)
        return card

    def _workspace_card(self) -> Card:
        paths, config = self.ctx.paths, self.ctx.config
        card = Card("Workspace")
        counts = self.ctx.store.counts()
        catalog = self.ctx.catalog.counts()
        self.workspace_values = KeyValues((
            ("Workspace", paths.workspace),
            ("Lab database", paths.lab_db),
            ("Schema", f"version {self.ctx.store.db.version} · {counts['experiments']} experiments · "
                       f"{counts['runs']} runs"),
            ("Catalogue", f"{catalog['datasets']} datasets · {catalog['models']} models · "
                          f"{catalog['plugins']} plugins · {catalog['runners']} runners"),
            ("MLflow", mlflow_tracking_uri(config, paths)),
            ("Logs", paths.logs / "app.log"),
            ("Settings", paths.settings_file),
            ("Python", platform.python_version()),
            ("Qt / PySide6", f"{qVersion()} / {PYSIDE_VERSION}"),
        ))
        card.add(self.workspace_values)
        buttons = QHBoxLayout()
        for title, page_id in (("Documentation", "documentation"), ("Settings", "settings")):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, target=page_id: self.ctx.navigate(target))
            buttons.addWidget(button)
        buttons.addStretch(1)
        card.add(buttons)
        return card

    def _problems_card(self, problems: list[str]) -> Card:
        card = Card("Catalogue problems", "these entries were skipped")
        card.add_head_widget(Pill(f"{len(problems)}", "fail"))
        for problem in problems:
            card.add(label(problem, "Mono", wrap=True, selectable=True))
        return card
