"""The page shown for anything not built yet: when it arrives and what it will do."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QWidget

from ...navigation import PageSpec
from ...services.context import AppContext
from ..widgets import Card, Page, Pill, label

__all__ = ["PlaceholderPage", "status_text"]


def status_text(spec: PageSpec) -> str:
    """``Planned for v0.2``, ``Planned for v0.1 · build phase 4``, ``Scope not decided``."""
    if spec.planned_for is None:
        return "Available"
    if spec.planned_for == "TBD":
        return "Planned · scope not decided"
    text = f"Planned for {spec.planned_for}"
    return f"{text} · build phase {spec.build_phase}" if spec.build_phase else text


class PlaceholderPage(Page):
    def __init__(self, ctx: AppContext, spec: PageSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec.title, status_text(spec).lower(), parent)
        self.spec = spec

        card = Card("Not available yet")
        self.pill = Pill(status_text(spec), "planned")
        card.add_head_widget(self.pill)
        card.add(label(spec.summary, "BodyStrong", wrap=True))
        if spec.features:
            card.add(label("What this page will do", "CardCaption"))
            for feature in spec.features:
                row = QHBoxLayout()
                row.setSpacing(8)
                row.addWidget(label("–", "Bullet"))
                row.addWidget(label(feature, "Body", wrap=True), 1)
                card.add(row)
        card.add(label("Nothing on this page works yet, so it has no controls. The build plan "
                       "is in CLAUDE.md, under Documentation.", "CardCaption", wrap=True))
        self.body.addWidget(card)
        self.body.addStretch(1)
