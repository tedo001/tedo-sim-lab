"""Pieces the research labs share: preset experiments (Run, Open in builder), the datasets of a lab
with their state, and pretrained models from the Hugging Face Hub (licence, size, Download), plus a
chooser of the downloaded ones."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Task
from core.model_registry.cards import ModelCard
from labs.common.remote import RemoteItem

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, FlowGrid, Pill, label
from .builder.form import task_title
from .catalog_common import licence_pill

__all__ = ["DownloadedModels", "LabDatasets", "PresetGrid", "PretrainedModels", "hub_repo_of"]

_STATE = {"ready": ("On disk", "ok"), "not_downloaded": ("Not downloaded", "warn"), "planned": ("Planned",
                                                                                                "planned"),
          "catalog_only": ("Catalogue", "planned"), "not_installed": ("Unavailable", "fail")}


def hub_repo_of(card: ModelCard) -> str | None:
    return card.hub_weights[0].hub_repo if card.hub_weights else None


class PresetGrid(Card):
    """Preset experiments as tiles: title, summary, cost; Run (disabled with the reason) and Open in
    builder."""

    def __init__(self, ctx: AppContext, presets: Iterable[Any], caption: str = "") -> None:
        super().__init__("Start from a preset", caption)
        self.ctx = ctx
        self.presets = list(presets)
        self.buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        tiles = []
        for preset in self.presets:
            tile = QWidget()
            box = QVBoxLayout(tile)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            box.addWidget(label(preset.title, "BodyStrong", wrap=True))
            box.addWidget(label(preset.summary, "Body", wrap=True))
            cost = getattr(preset, "cost", "")
            box.addWidget(label(task_title(preset.make().task) + (f" · {cost}" if cost else ""),
                                "CardCaption",
                                wrap=True))
            run = QPushButton(icon("play"), "Run")
            run.clicked.connect(lambda _=False, p=preset: self.run(p))
            edit = QPushButton("Open in builder")
            edit.clicked.connect(lambda _=False, p=preset: ctx.experiments.open_in_builder(p.make(),
                                                                                           ctx.navigate))
            row = QHBoxLayout()
            row.addWidget(run)
            row.addWidget(edit)
            row.addStretch(1)
            box.addLayout(row)
            box.addStretch(1)
            self.buttons[preset.id] = (run, edit)
            tiles.append(tile)
        self.add(FlowGrid(tiles, 260, fewest=1))

    def refresh(self) -> None:
        for preset in self.presets:
            run, _edit = self.buttons[preset.id]
            problems = self.ctx.experiments.problems(preset.make())
            run.setEnabled(not problems)
            run.setToolTip("\n".join(problems))

    def run(self, preset: Any) -> str | None:
        spec = preset.make()
        if self.ctx.experiments.problems(spec):
            return None
        run_id = self.ctx.experiments.launch(spec)
        self.ctx.navigate("training")
        return run_id


class LabDatasets(Card):
    """The datasets a lab works with: size, licence, state; opens the chosen one in the Dataset Hub."""

    def __init__(self, ctx: AppContext, keep: Callable[[Any], bool], caption: str = "") -> None:
        super().__init__("Datasets", caption or "download them in the Dataset Hub", padded=False)
        self.ctx, self.keep = ctx, keep
        self.ids: list[str] = []
        hub = QPushButton(icon("database"), "Open in Dataset Hub")
        hub.clicked.connect(self.open_hub)
        self.add_head_widget(hub)
        self.table = DataTable(("Dataset", "Size", "Licence", "State"), mono_columns=(1,), stretch_column=0)
        self.add(self.table)
        ctx.downloads.finished.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        registry = self.ctx.catalog.datasets
        self.table.clear_rows()
        self.ids = []
        for card in registry.all():
            if not self.keep(card):
                continue
            self.ids.append(card.id)
            text, tone = _STATE.get(registry.status(card.id), ("Unknown", "planned"))
            self.table.add_row((card.name, card.size or "—", licence_pill(card.license), Pill(text, tone)))

    def open_hub(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        card_id = self.ids[rows[0].row()] if rows and self.ids else (self.ids[0] if self.ids else None)
        hub = self.ctx.show_page("dataset_hub")
        if card_id is not None:
            hub.show_view("catalogue").select(card_id)


class PretrainedModels(Card):
    """Pretrained models for some tasks: licence of the weights, size, state and a Download button
    (through the Hugging Face plugin: licence checked, gated repositories refused)."""

    changed = Signal()

    def __init__(self, ctx: AppContext, tasks: Iterable[Task], caption: str = "") -> None:
        super().__init__("Pretrained models",
                         caption or "downloaded once into this workspace's models/ folder",
                         padded=False)
        self.ctx, self.tasks = ctx, set(tasks)
        self.jobs: dict[str, str] = {}  # job id → card id
        self.table = DataTable(("Model", "Size", "Weights' licence", "State", ""), stretch_column=0)
        self.add(self.table)
        self.message = label("", "CardCaption", wrap=True)
        self.add(self.message)
        ctx.plugins.finished.connect(self._finished)
        self.refresh()

    def cards(self) -> list[ModelCard]:
        return [card for card in self.ctx.catalog.models.all() if card.hub_weights
                and set(card.tasks) & self.tasks]

    def refresh(self) -> None:
        models = self.ctx.catalog.models
        self.table.clear_rows()
        busy = set(self.jobs.values())
        for card in self.cards():
            weights = card.hub_weights[0]
            size = f"{weights.size_mb:g} MB" if weights.size_mb else "—"
            here = models.downloaded(card.id)
            state = Pill("Downloading", "info") if card.id in busy else \
                Pill("On disk", "ok") if here else Pill("Not downloaded", "warn")
            button = QPushButton(icon("download"), "Download" if not here else "Again")
            button.setToolTip(f"{weights.hub_repo} · {weights.license.name}")
            button.setEnabled(card.id not in busy)
            button.clicked.connect(lambda _=False, c=card: self.download(c))
            self.table.add_row((card.name, size, licence_pill(weights.license), state, button))

    def download(self, card: ModelCard) -> str:
        weights = card.hub_weights[0]
        item = RemoteItem("huggingface", str(weights.hub_repo), str(weights.hub_repo), "model",
                          f"https://huggingface.co/{weights.hub_repo}", licence=weights.license.name)
        job = self.ctx.plugins.download("huggingface", item)
        self.jobs[job] = card.id
        self.message.setText(f"Downloading {card.name} ({weights.hub_repo})…")
        self.refresh()
        return job

    def _finished(self, job_id: str, _name: str, action: str, _result: object, error: str) -> None:
        if action != "download" or job_id not in self.jobs:
            return
        card = self.ctx.catalog.models.get(self.jobs.pop(job_id))
        self.message.setText(f"{card.name}: {error}" if error else f"{card.name} is ready.")
        self.refresh()
        self.changed.emit()


class DownloadedModels(QComboBox):
    """The downloaded pretrained models for some tasks; ``repo`` is the chosen Hub repository."""

    def __init__(self, ctx: AppContext, tasks: Iterable[Task]) -> None:
        super().__init__()
        self.ctx, self.tasks = ctx, set(tasks)
        self.setMinimumWidth(220)
        self.setMaximumWidth(380)
        self.refresh()

    def refresh(self) -> None:
        current = self.currentData()
        self.blockSignals(True)
        self.clear()
        models = self.ctx.catalog.models
        for card in models.all():
            if card.hub_weights and set(card.tasks) & self.tasks and models.downloaded(card.id):
                self.addItem(card.name, hub_repo_of(card))
        if not self.count():
            self.addItem("Download a model below first", None)
        self.setCurrentIndex(max(self.findData(current), 0))
        self.blockSignals(False)

    @property
    def repo(self) -> str | None:
        return self.currentData()
