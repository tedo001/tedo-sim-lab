"""Explore a dataset in 3-D: its rows as points along three columns or along the first three
principal components, coloured by class (or by a numeric target) and turned with the mouse.
Tables from the catalogue (scikit-learn's own and imported CSV files) and MNIST-style image sets
once they are downloaded; reading and the components run in the background."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtWidgets import QComboBox, QFormLayout, QSpinBox, QVBoxLayout, QWidget

from labs.common.explore import PointTable, explorable, image_points, principal_components, tabular_points

from ...services.context import AppContext
from ..widgets import Card, ResponsiveRow, label
from ..widgets.plots import BarList
from ..widgets.scatter3d import GROUPS, ScatterView

__all__ = ["DatasetExplorer"]

log = logging.getLogger("tedo.datasets")

AXES = {"components": "Principal components", "columns": "Columns"}


def _combo(width: int = 160) -> QComboBox:
    combo = QComboBox()
    combo.setMinimumWidth(width)
    return combo


class DatasetExplorer(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.table: PointTable | None = None
        self.scores: np.ndarray | None = None
        self.share: np.ndarray | None = None
        self._job: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls = Card("Explore in 3-D", "rows as points; drag to turn")
        form = QFormLayout()
        self.dataset = _combo(180)
        self.rows = QSpinBox()
        self.rows.setRange(200, 20000)
        self.rows.setSingleStep(500)
        self.rows.setValue(5000)
        self.rows.setSuffix(" rows at most")
        self.axes_box = _combo()
        for key, title in AXES.items():
            self.axes_box.addItem(title, key)
        self.columns = [_combo() for _ in range(3)]
        self.colour = _combo()
        self.highlight = _combo()
        for title, widget in (("Dataset", self.dataset), ("Axes", self.axes_box), ("X", self.columns[0]),
                              ("Y", self.columns[1]), ("Z (up)", self.columns[2]), ("Colour", self.colour),
                              ("Highlight", self.highlight), ("Sample", self.rows)):
            form.addRow(title, widget)
        controls.add(form)
        self.status = label("", "CardCaption", wrap=True)
        controls.add(self.status)

        self.view = ScatterView("")
        self.view.setMinimumHeight(440)
        shown = Card("Points")
        shown.add(self.view)
        layout.addWidget(ResponsiveRow([(controls, 0), (shown, 1)], breakpoint=900))

        self.components_card = Card("Principal components", "the share of the variance along each direction")
        self.components = BarList()
        self.components_card.add(self.components)
        self.components_note = label("", "CardCaption", wrap=True)
        self.components_card.add(self.components_note)
        layout.addWidget(self.components_card)

        self.dataset.currentIndexChanged.connect(lambda _i: self.load())
        self.rows.editingFinished.connect(self.load)
        self.axes_box.currentIndexChanged.connect(lambda _i: self._axes_changed())
        for combo in (*self.columns, self.colour):
            combo.currentIndexChanged.connect(lambda _i: self.draw())
        self.highlight.currentIndexChanged.connect(lambda index: self.view.set_highlight(index))
        ctx.jobs.job_finished.connect(self._loaded)
        ctx.downloads.finished.connect(lambda *_: self.refresh())
        ctx.downloads.imported.connect(lambda *_: self.refresh())
        self.refresh()

    # Datasets -----------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """List the datasets that can be drawn; those not on disk yet say so."""
        registry = self.ctx.catalog.datasets
        current = self.dataset.currentData()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for card in registry.all():
            if card.adapter is None:
                continue
            try:
                adapter = registry.adapter(card.id)
            except (ImportError, AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
                continue
            if not explorable(adapter):
                continue
            here = adapter.local_status(registry.root) == "present"
            self.dataset.addItem(card.name if here else f"{card.name} (download it first)", card.id)
        index = self.dataset.findData(current)
        self.dataset.setCurrentIndex(index if index >= 0 else 0)
        self.dataset.blockSignals(False)
        if self.table is None or index < 0:
            self.load()

    def select(self, card_id: str) -> bool:
        index = self.dataset.findData(card_id)
        if index >= 0:
            self.dataset.setCurrentIndex(index)
        return index >= 0

    def load(self) -> None:
        card_id = self.dataset.currentData()
        if card_id is None:
            self.status.setText("No dataset here can be drawn as points yet.")
            return
        registry = self.ctx.catalog.datasets
        adapter = registry.adapter(card_id)
        if adapter.local_status(registry.root) != "present":
            self.status.setText(f"{adapter.card.name} is not on this computer: download it in the Catalogue.")
            self.table = None
            self.view.set_points(np.zeros((0, 3)))
            return
        limit, root = self.rows.value(), registry.root

        def work(cancel, progress):
            try:
                table = (tabular_points(adapter, root, limit=limit) if hasattr(adapter, "load_frame")
                         else image_points(adapter, root, limit=min(limit, 10000)))
            except (OSError, ImportError, KeyError, ValueError) as exc:  # a file that cannot be read
                log.info("Could not read %s as points: %s", adapter.card.name, exc)
                return card_id, None, str(exc)
            cancel.raise_if_cancelled()
            return card_id, table, principal_components(table.values, standardize=table.standardize)

        self.status.setText(f"Reading {adapter.card.name}…")
        self._job = self.ctx.jobs.submit_task(work, title=f"Read {adapter.card.name} as points", quiet=True)

    def _loaded(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.status.setText(f"Could not read it: {job.error or status}")
            return
        card_id, table, found = job.result
        if card_id != self.dataset.currentData():
            return
        if table is None:
            self.status.setText(f"Could not read it: {found}")
            return
        self.table, (self.scores, self.share) = table, found
        self._fill_choices()
        self.draw()

    # Drawing --------------------------------------------------------------------------------------------
    def _fill_choices(self) -> None:
        table = self.table
        for index, combo in enumerate(self.columns):
            combo.blockSignals(True)
            combo.clear()
            for name in table.columns:
                combo.addItem(name, name)
            combo.setCurrentIndex(min(index, max(len(table.columns) - 1, 0)))
            combo.blockSignals(False)
        self.axes_box.blockSignals(True)
        self.axes_box.setCurrentIndex(self.axes_box.findData("columns" if 0 < len(table.columns) <= 3
                                                             else "components"))
        self.axes_box.model().item(self.axes_box.findData("columns")).setEnabled(len(table.columns) > 0)
        self.axes_box.blockSignals(False)
        self.colour.blockSignals(True)
        self.colour.clear()
        if table.groups is not None:
            self.colour.addItem("By class", "groups")
        if table.numbers is not None:
            self.colour.addItem(f"By {table.number_name}", "numbers")
        self.colour.addItem("One colour", "none")
        self.colour.blockSignals(False)
        self.highlight.blockSignals(True)
        self.highlight.clear()
        for name in table.group_names:
            self.highlight.addItem(name)
        self.highlight.blockSignals(False)
        self._axes_changed(draw=False)
        share = self.share[:10]
        self.components.set_rows([(f"PC {i + 1}", float(value)) for i, value in enumerate(share)])
        first = float(self.share[:3].sum()) if len(self.share) else 0.0
        scale = "" if table.standardize else " (pixels left in their own scale)"
        self.components_note.setText(f"The first three components hold {first:.0%} of the variance{scale}.")

    def _axes_changed(self, draw: bool = True) -> None:
        columns = self.axes_box.currentData() == "columns"
        for combo in self.columns:
            combo.setEnabled(columns)
        if draw:
            self.draw()

    def draw(self) -> None:
        table = self.table
        if table is None:
            return
        if self.axes_box.currentData() == "columns" and table.columns:
            picked = [combo.currentIndex() for combo in self.columns]
            points = table.values[:, picked]
            axes = tuple(table.columns[i] for i in picked)
        else:
            points = self.scores[:, :3]
            axes = tuple(f"PC {i + 1} ({share:.0%})" for i, share in enumerate(self.share[:3]))
        colour = self.colour.currentData()
        groups = table.groups if colour == "groups" else None
        numbers = table.numbers if colour == "numbers" else None
        self.view.title = table.name
        self.view.set_points(points, groups=groups, names=table.group_names if groups is not None else (),
                             values=numbers, axes=axes, value_label=table.number_name,
                             highlight=max(self.highlight.currentIndex(), 0))
        self.highlight.setEnabled(groups is not None and len(table.group_names) > len(GROUPS))
        text = f"{table.rows:,} rows"
        if table.note:
            text += f" · {table.note}"
        if groups is not None and len(table.group_names) > len(GROUPS):
            text += " · one class at a time is coloured (Highlight); the rest are grey"
        self.status.setText(text)
