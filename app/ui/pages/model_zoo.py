"""Model Zoo: every model card (vision, OCR, language, audio, classical) with its code licence,
the separate licence of any pretrained weights, size and hardware needs, and whether the lab
can build it yet. Third-party models must be permissively licensed; the ones kept out by that
policy are listed with the reason."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from core.common.vocab import Framework, Task
from labs.common.remote import RemoteItem

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, label
from .builder.form import task_title
from .catalog_common import FilterRow, licence_pill, status_pill

__all__ = ["ModelZooPage", "excluded_tools"]

#: Where a model is put to use, by task.
_LAB_OF = {Task.TEXT_GENERATION: "nlp_llm", Task.TEXT_CLASSIFICATION: "nlp_llm",
           Task.SPEECH_RECOGNITION: "audio",
           Task.ZERO_SHOT_CLASSIFICATION: "multimodal", Task.IMAGE_CAPTIONING: "multimodal",
           Task.OBJECT_DETECTION: "testing", Task.IMAGE_CLASSIFICATION: "computer_vision",
           Task.QUANTUM_CLASSIFICATION: "quantum_ml", Task.TABULAR_CLASSIFICATION: "classical_ml",
           Task.TABULAR_REGRESSION: "classical_ml", Task.CLUSTERING: "classical_ml",
           Task.DIMENSIONALITY_REDUCTION: "classical_ml", Task.AUDIO_CLASSIFICATION: "audio"}


def excluded_tools(code_root: Path) -> list[dict]:
    path = code_root / "configs" / "excluded_tools.yaml"
    if not path.is_file():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("excluded", [])


def _params(count: int | None) -> str:
    if count is None:
        return "—"
    return f"{count / 1e6:.1f} M" if count >= 1e6 else f"{count / 1e3:.1f} k" if count >= 1e3 else str(count)


class ModelZooPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Model Zoo", "permissively licensed models, their weights' terms and hardware needs",
                         parent)
        self.ctx = ctx
        self.ids: list[str] = []
        self.selected: str | None = None
        self.filters = FilterRow("Search models, tasks, tags…", [("Any framework", None)] + [
            (framework.value, framework) for framework in Framework])
        self.filters.connect(self.refresh)
        self.body.addWidget(self.filters)

        listing = Card("Models", padded=False)
        self.count = label("", "CardCaption")
        listing.add_head_widget(self.count)
        self.table = DataTable(("Model", "Framework", "Params", "Licence", "Status"), mono_columns=(2,),
                               stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        listing.add(self.table)
        self.body.addWidget(listing)

        self.detail = Card("")
        self.head_pills = QHBoxLayout()
        holder = QWidget()
        holder.setLayout(self.head_pills)
        self.head_pills.setContentsMargins(0, 0, 0, 0)
        self.detail.add_head_widget(holder)
        self.description = label("", "Body", wrap=True)
        self.values = KeyValues((("Architecture", ""), ("Tasks", ""), ("Source", ""), ("Code licence", ""),
                                 ("Parameters", ""), ("Input", ""), ("Hardware", ""), ("Extra packages", ""),
                                 ("Origin", "")))
        self.notes = label("", "Body", wrap=True)
        self.weights = DataTable(("Pretrained weights", "Licence", "Size", "Access"), mono_columns=(0, 2),
                                 stretch_column=0)
        self.state = label("", "CardCaption", wrap=True)
        buttons = QHBoxLayout()
        self.source_button = QPushButton(icon("external-link"), "Open source page")
        self.source_button.clicked.connect(self._open_source)
        self.download_button = QPushButton(icon("download"), "Download weights")
        self.download_button.setToolTip("Through the Hugging Face plugin: licence checked, gated models "
                                        "refused")
        self.download_button.clicked.connect(self._download)
        self.use_button = QPushButton(icon("test-tube"), "Use it")
        self.use_button.clicked.connect(self._use)
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.source_button)
        buttons.addStretch(1)
        self._downloads: dict[str, str] = {}
        ctx.plugins.finished.connect(self._downloaded)
        for widget in (self.description, self.values, self.notes, self.weights, self.state):
            self.detail.add(widget)
        self.detail.add(buttons)
        self.body.addWidget(self.detail)

        excluded = Card("Not in the lab", "kept out by the licence policy (permissive licences only)",
                        padded=False)
        self.excluded = DataTable(("Model or tool", "Licence", "Why"), stretch_column=2)
        for item in excluded_tools(ctx.paths.code_root):
            self.excluded.add_row((item["name"], item["licence"], item["reason"]))
        excluded.add(self.excluded)
        self.body.addWidget(excluded)
        self.body.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        registry = self.ctx.catalog.models
        cards = [card for card in registry.all()
                 if self.filters.matches(card.search_text() + " " + card.architecture, card.framework,
                                         card.license.category)]
        self.ids = [card.id for card in cards]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for card in cards:
            self.table.add_row((card.name, card.framework.value, _params(card.params),
                                licence_pill(card.license),
                                status_pill(registry.status(card.id), card.planned_for)))
        self.table.blockSignals(False)
        self.count.setText(f"{len(cards)} of {len(registry)}")
        if self.selected not in self.ids:
            self.selected = self.ids[0] if self.ids else None
        if self.selected is not None:
            self.table.blockSignals(True)
            self.table.selectRow(self.ids.index(self.selected))
            self.table.blockSignals(False)
        self._show()

    def select(self, card_id: str) -> None:
        self.filters.reset()
        self.selected = card_id
        self.refresh()

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.ids[rows[0].row()]
            self._show()

    def _show(self) -> None:
        self.detail.setVisible(self.selected is not None)
        if self.selected is None:
            return
        registry = self.ctx.catalog.models
        card = registry.get(self.selected)
        status, detail = registry.status_detail(card.id)
        self.detail.set_title(card.name)
        while self.head_pills.count():
            widget = self.head_pills.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.head_pills.addWidget(licence_pill(card.license))
        self.head_pills.addWidget(status_pill(status, card.planned_for))
        self.description.setText(card.description)
        self.description.setVisible(bool(card.description))
        self.values.set_value("Architecture", card.architecture)
        self.values.set_value("Tasks", ", ".join(task_title(t) for t in card.tasks))
        self.values.set_value("Source", card.source_url)
        self.values.set_value("Code licence", card.license.name)
        self.values.set_value("Parameters", f"{card.params:,}" if card.params else "—")
        self.values.set_value("Input", card.input_spec or "—")
        self.values.set_value("Hardware", f"a GPU with {card.min_vram_gb:g} GB or more" if card.min_vram_gb
                              else "runs on a CPU")
        self.values.set_value("Extra packages", ", ".join(card.requires) or (
            "transformers (in the experiment Python)" if card.framework == Framework.HUGGINGFACE else "none"))
        self.values.set_value("Origin", "written for this lab" if card.origin == "builtin" else "third party")
        self.notes.setText(card.license.notes)
        self.notes.setVisible(bool(card.license.notes))
        self.weights.clear_rows()
        for weights in card.weights:
            size = f"{weights.size_mb:g} MB" if weights.size_mb else "—"
            self.weights.add_row((weights.id, licence_pill(weights.license), size,
                                  weights.access_requirements or "no account needed"))
        self.weights.setVisible(bool(card.weights))
        self.state.setText(detail)
        hub = bool(card.hub_weights)
        busy = card.id in self._downloads.values()
        self.download_button.setVisible(hub)
        self.download_button.setEnabled(not busy)
        self.download_button.setText("Downloading…" if busy else "Download again" if status == "ready" else
                                     "Download weights")
        self.use_button.setVisible(status == "ready" and any(task in _LAB_OF for task in card.tasks))

    def _download(self) -> None:
        if self.selected is None:
            return
        weights = self.ctx.catalog.models.get(self.selected).hub_weights[0]
        item = RemoteItem("huggingface", str(weights.hub_repo), str(weights.hub_repo), "model",
                          f"https://huggingface.co/{weights.hub_repo}", licence=weights.license.name)
        self._downloads[self.ctx.plugins.download("huggingface", item)] = self.selected
        self._show()

    def _downloaded(self, job_id: str, _name: str, action: str, _result: object, error: str) -> None:
        if action == "download" and job_id in self._downloads:
            card_id = self._downloads.pop(job_id)
            self.refresh()
            if card_id == self.selected and error:
                self.state.setText(f"Download failed: {error}")

    def _use(self) -> None:
        """Where the model is used: its lab, or Model Testing for detectors, captioners and the like."""
        if self.selected is None:
            return
        card = self.ctx.catalog.models.get(self.selected)
        lab = next((_LAB_OF[task] for task in card.tasks if task in _LAB_OF), None)
        if lab == "testing":
            self.ctx.show_page("testing").open_model(card.id)
        elif lab:
            self.ctx.navigate(lab)

    def _open_source(self) -> None:
        if self.selected is not None:
            QDesktopServices.openUrl(QUrl(self.ctx.catalog.models.get(self.selected).source_url))
