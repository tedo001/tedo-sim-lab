"""Dataset Hub: every dataset the lab knows with its licence and access terms (downloads only
where they allow it), the Ontology Explorer (ImageNet classes in WordNet) and the COCO
annotation inspector, one view at a time."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...services.context import AppContext
from ..widgets import Page, Segmented
from .coco_view import CocoView
from .dataset_catalog import DatasetCatalog
from .ontology_view import OntologyView

__all__ = ["DatasetHubPage"]


class DatasetHubPage(Page):
    VIEWS = (("catalogue", "Catalogue"), ("ontology", "Ontology Explorer"), ("coco", "COCO inspector"))

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Dataset Hub", "licences, sources and access; download only where allowed", parent)
        self.ctx = ctx
        self.switch = Segmented(self.VIEWS)
        self.body.addWidget(self.switch)
        # One view visible at a time; a plain column (not a stack) so each view keeps its own height.
        self.holder = QWidget()
        self.column = QVBoxLayout(self.holder)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.views: dict[str, QWidget] = {}
        self._factories = {"catalogue": DatasetCatalog, "ontology": OntologyView, "coco": CocoView}
        self.body.addWidget(self.holder)
        self.body.addStretch(1)
        self.switch.changed.connect(self.show_view)
        self.show_view("catalogue")

    def show_view(self, key: str) -> QWidget:
        """Switch to ``key``, building that view the first time it is shown."""
        if key not in self.views:
            self.views[key] = self._factories[key](self.ctx)
            self.column.addWidget(self.views[key])
        if self.switch.current != key:
            self.switch.buttons[key].setChecked(True)
        for name, view in self.views.items():
            view.setVisible(name == key)
        return self.views[key]
