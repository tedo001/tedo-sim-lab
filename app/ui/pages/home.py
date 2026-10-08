"""Home: the dashboard. Live resources, the active job, recent work, the catalogue,
MLflow, the workspace, and what this build can do so far."""

from __future__ import annotations

import platform

from PySide6 import __version__ as PYSIDE_VERSION
from PySide6.QtCore import qVersion
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ... import __version__
from ...navigation import NAV, SECTIONS
from ...services.context import AppContext
from ..widgets import Card, DataTable, KeyValues, Page, Pill, ResponsiveRow, StatTile, label
from .home_cards import ActiveJobCard, CatalogueCard, JobHistoryCard, MlflowCard, RecentExperimentsCard
from .placeholder import status_text
from .resources import ResourceCharts, ResourceStats

__all__ = ["HomePage"]

_SECTION_NAMES = dict(SECTIONS, home="Home", system="System")


def _tone(planned_for: str | None) -> str:
    if planned_for is None:
        return "ok"
    return "info" if planned_for == "v0.1" else "planned"


def _row(*widgets: tuple[QWidget, int]) -> ResponsiveRow:
    return ResponsiveRow(list(widgets))


class HomePage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Home", f"v{__version__} · workspace {ctx.paths.workspace}", parent)
        self.ctx = ctx
        self.jobs_tile = StatTile("Jobs")
        self.experiments_tile = StatTile("Experiments")
        self.stats = ResourceStats(ctx.hardware, extra={"jobs": self.jobs_tile,
                                                        "experiments": self.experiments_tile})
        self.body.addWidget(self.stats)

        resources = Card("Resources", "last two minutes")
        resources.add(ResourceCharts(ctx.hardware, columns=2))
        self.active_job = ActiveJobCard(ctx)
        self.body.addWidget(_row((resources, 2), (self.active_job, 1)))

        self.experiments = RecentExperimentsCard(ctx)
        self.history = JobHistoryCard(ctx)
        self.body.addWidget(_row((self.experiments, 1), (self.history, 1)))

        self.body.addWidget(_row((CatalogueCard(ctx), 1), (MlflowCard(ctx), 1),
                                 (self._workspace_card(), 1)))
        if ctx.catalog.errors:
            self.body.addWidget(self._problems_card(ctx.catalog.errors))
        self.body.addWidget(self._status_card())
        self.body.addStretch(1)

        for signal in (ctx.jobs.job_queued, ctx.jobs.job_started, ctx.jobs.job_finished):
            signal.connect(lambda *_: self.refresh_counts())
        self.refresh_counts()

    def refresh_counts(self) -> None:
        active = self.ctx.jobs.active()
        running = sum(job.status == "running" for job in active)
        self.jobs_tile.set(str(running), f"running · {len(active) - running} queued")
        counts = self.ctx.store.counts()
        self.experiments_tile.set(str(counts["experiments"]), f"{counts['runs']} runs")
        self.experiments.refresh()

    def _workspace_card(self) -> Card:
        paths = self.ctx.paths
        card = Card("Workspace")
        self.workspace_values = KeyValues((
            ("Workspace", paths.workspace),
            ("Lab database", paths.lab_db),
            ("Schema", f"version {self.ctx.store.db.version}"),
            ("Logs", paths.logs / "app.log"),
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

    def _status_card(self) -> Card:
        built = sum(spec.built for spec in NAV)
        card = Card("Build status", f"{built} of {len(NAV)} pages live", padded=False)
        self.status_table = DataTable(("Page", "Section", "Status"), stretch_column=1)
        for spec in NAV:
            self.status_table.add_row((spec.title, _SECTION_NAMES[spec.section],
                                       Pill(status_text(spec), _tone(spec.planned_for))))
        card.add(self.status_table)
        return card

    def _problems_card(self, problems: list[str]) -> Card:
        card = Card("Catalogue problems", "these entries were skipped")
        card.add_head_widget(Pill(f"{len(problems)}", "fail"))
        for problem in problems:
            card.add(label(problem, "Mono", wrap=True, selectable=True))
        return card
