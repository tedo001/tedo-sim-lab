"""What Ctrl+K finds: every page, dataset card, model card and recent run, as
(label shown in the list, key). Keys are ``page:<id>``, ``dataset:<id>``, ``model:<id>`` and
``run:<id>``; :meth:`app.main_window.MainWindow.open_entry` opens them."""

from __future__ import annotations

from ...navigation import NAV
from ...services.context import AppContext

__all__ = ["search_entries"]


def search_entries(ctx: AppContext, *, runs: int = 200) -> list[tuple[str, str]]:
    entries = [(spec.title, f"page:{spec.id}") for spec in NAV]
    entries += [(f"Dataset · {card.name}", f"dataset:{card.id}") for card in ctx.catalog.datasets.all()]
    entries += [(f"Model · {card.name}", f"model:{card.id}") for card in ctx.catalog.models.all()]
    entries += [(f"Run · {view.name} · {view.id[:8]}", f"run:{view.id}")
                for view in ctx.experiments.views(limit=runs)]
    return entries
