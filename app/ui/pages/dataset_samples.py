"""The Dataset Hub's preview of a downloaded dataset: a few texts with their labels, spectrograms or
images with their labels, or the first rows of a table. Read in a quiet background task."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from labs.common.samples import previewable, samples

from ...services.context import AppContext
from ..widgets import Card, DataTable, FlowGrid, label

__all__ = ["DatasetSamples", "pixmap_of"]

log = logging.getLogger("tedo.ui.datasets")
TILE = 96


def pixmap_of(array: np.ndarray, side: int = TILE) -> QPixmap:
    """A greyscale (h × w) or colour (h × w × 3) uint8 array as a pixmap, scaled without smoothing."""
    data = np.ascontiguousarray(array, dtype=np.uint8)
    if data.ndim == 2:
        image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0],
                       QImage.Format.Format_Grayscale8)
    else:
        image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.FastTransformation)


def _tile(name: str, array: np.ndarray) -> QWidget:
    tile = QWidget()
    column = QVBoxLayout(tile)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    picture = QLabel()
    picture.setPixmap(pixmap_of(array))
    picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(picture)
    column.addWidget(label(name, "Mono"), 0, Qt.AlignmentFlag.AlignHCenter)
    return tile


class DatasetSamples(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Examples", "a few items picked at random", parent=parent)
        self.ctx = ctx
        self._job: str | None = None
        self.card_id: str | None = None
        self.status = label("", "CardCaption", wrap=True)
        self.add(self.status)
        self.holder = QWidget()
        self.column = QVBoxLayout(self.holder)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.add(self.holder)
        ctx.jobs.job_finished.connect(self._loaded)

    def _clear(self) -> None:
        while self.column.count():
            widget = self.column.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def set_card(self, card_id: str | None) -> None:
        """Show examples of ``card_id`` when it is on disk; hide otherwise."""
        self.card_id = card_id
        self._clear()
        registry = self.ctx.catalog.datasets
        try:
            adapter = registry.adapter(card_id) if card_id else None
        except (ImportError, AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            adapter = None
        if adapter is None or not previewable(adapter) or adapter.local_status(registry.root) != "present":
            self.setVisible(False)
            return
        self.setVisible(True)
        self.status.setText(f"Reading {adapter.card.name}…")
        root = registry.root

        def work(cancel, progress):  # noqa: ANN001, ARG001 (task signature)
            try:
                return card_id, samples(adapter, root), ""
            except (OSError, ImportError, KeyError, ValueError) as exc:
                log.info("Could not read examples of %s: %s", adapter.card.name, exc)
                return card_id, None, str(exc)

        self._job = self.ctx.jobs.submit_task(work, title=f"Examples of {adapter.card.name}", quiet=True)

    def _loaded(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.status.setText(f"Could not read it: {job.error or status}")
            return
        card_id, found, error = job.result
        if card_id != self.card_id:
            return
        if found is None:
            self.status.setText(f"Could not read it: {error}")
            return
        self._clear()
        total = f"{found.get('total', 0):,}"
        if found["kind"] == "text":
            self.status.setText(f"{len(found['rows'])} of {total} texts")
            table = DataTable(("Label", "Text"), stretch_column=1)
            for name, text in found["rows"]:
                table.add_row((name, text))
            self.column.addWidget(table)
        elif found["kind"] == "table":
            self.status.setText(f"The first {len(found['rows'])} of {total} rows")
            table = DataTable(tuple(found["columns"]), stretch_column=0)
            for row in found["rows"]:
                table.add_row(tuple(row))
            self.column.addWidget(table)
        else:
            self.status.setText(f"{len(found['images'])} of {total} · {found['caption']}")
            self.column.addWidget(FlowGrid([_tile(name, array) for name, array in found["images"]], TILE + 8,
                                           fewest=2))
