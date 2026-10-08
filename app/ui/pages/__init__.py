"""Built pages by id. Every page in :data:`app.navigation.NAV` with
``planned_for=None`` must appear here; everything else gets a placeholder.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from ...navigation import page
from ...services.context import AppContext
from .documentation import DocumentationPage
from .home import HomePage
from .placeholder import PlaceholderPage
from .settings import SettingsPage

__all__ = ["PAGE_FACTORIES", "build_page"]

PAGE_FACTORIES: dict[str, Callable[[AppContext], QWidget]] = {
    "home": HomePage,
    "documentation": DocumentationPage,
    "settings": SettingsPage,
}


def build_page(page_id: str, ctx: AppContext) -> QWidget:
    """The page for ``page_id``: its factory if built, otherwise a placeholder."""
    spec = page(page_id)
    factory = PAGE_FACTORIES.get(page_id)
    return factory(ctx) if factory else PlaceholderPage(ctx, spec)
