"""The dataset and model steps of the Experiment Builder: what can be used for the task,
whether it is on disk, its licence, and the download or acknowledgement it needs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.common.licensing import download_policy
from core.common.vocab import Task

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Pill, label

__all__ = ["DatasetChooser", "DatasetDownload", "ModelChooser"]

_STATUS = {"ready": ("On disk", "ok"), "not_downloaded": ("Not downloaded", "warn"),
           "planned": ("Planned", "planned"), "not_installed": ("Unavailable", "fail")}


def _combo() -> QComboBox:
    combo = QComboBox()
    combo.setMinimumWidth(220)
    combo.setMaximumWidth(420)
    return combo


class DatasetDownload(QWidget):
    """Status, licence, acknowledgement, Download button and progress for one dataset."""

    changed = Signal()

    def __init__(self, ctx: AppContext, *, show_status: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.card_id: str | None = None
        self._asks = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.status = Pill("", "planned")
        self.status.setVisible(show_status)
        self.licence = label("", "CardCaption", wrap=True)
        self.terms = label("", "Body", wrap=True)
        self.acknowledge = QCheckBox("I accept these terms")
        self.acknowledge.toggled.connect(self._refresh_button)
        actions = QHBoxLayout()
        self.download = QPushButton(icon("database"), "Download")
        self.download.clicked.connect(self._download)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.hide()
        self.message = label("", "Mono", wrap=True)
        actions.addWidget(self.download)
        actions.addWidget(self.message, 1)
        for widget in (self.licence, self.terms, self.acknowledge):
            layout.addWidget(widget)
        layout.addLayout(actions)
        layout.addWidget(self.progress)
        ctx.downloads.progress.connect(self._on_progress)
        ctx.downloads.finished.connect(self._on_finished)

    def set_card(self, card_id: str | None) -> None:
        self.card_id = card_id
        self.refresh()

    @property
    def ready(self) -> bool:
        return self.card_id is not None and self.ctx.catalog.datasets.status(self.card_id) == "ready"

    def refresh(self) -> None:
        card_id = self.card_id
        if card_id is None:
            self.status.setText("None available")
            self.status.set_tone("planned")
            for widget in (self.licence, self.terms, self.acknowledge, self.download):
                widget.hide()
            self.changed.emit()
            return
        card = self.ctx.catalog.datasets.get(card_id)
        status = self.ctx.catalog.datasets.status(card_id)
        text, tone = _STATUS.get(status, (status, "planned"))
        self.status.setText(text)
        self.status.set_tone(tone)
        decision = download_policy(card)
        self.licence.setText(f"Licence: {card.license.name} ({card.license.label.lower()})."
                             + (f" {card.license.notes}" if card.license.notes else ""))
        self.licence.show()
        missing = status != "ready"
        asks = missing and bool(decision.acknowledgement)
        self._asks = asks
        self.acknowledge.setVisible(asks)
        self.terms.setVisible(asks)
        self.terms.setText(decision.acknowledgement or "")
        self.download.setVisible(missing and decision.allowed)
        if missing and not decision.allowed:
            self.message.setText(decision.reason)
        elif not self.ctx.downloads.active(card_id):
            self.message.setText("")
        self._refresh_button()
        self.changed.emit()

    def _refresh_button(self) -> None:
        card_id = self.card_id
        busy = card_id is not None and self.ctx.downloads.active(card_id)
        needs = self._asks and not self.acknowledge.isChecked()
        self.download.setEnabled(not busy and not needs)
        self.download.setToolTip("Accept the terms above first" if needs else "")

    def _download(self) -> None:
        if self.card_id is not None:
            self.ctx.downloads.start(self.card_id, acknowledged=self.acknowledge.isChecked())
            self.progress.setRange(0, 0)
            self.progress.show()
            self._refresh_button()

    def _on_progress(self, card_id: str, fraction: float, message: str) -> None:
        if card_id != self.card_id:
            return
        self.progress.show()
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        self.message.setText(message)

    def _on_finished(self, card_id: str, status: str, error: str) -> None:
        if card_id == self.card_id:
            self.progress.hide()
            words = {"failed": error, "completed": ""}
            self.message.setText(words.get(status, "Cancelled"))
            self.refresh()


class DatasetChooser(QWidget):
    """Which dataset, with its download controls."""

    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        self.combo = _combo()
        self.combo.currentIndexChanged.connect(lambda _: self.panel.set_card(self.card_id))
        self.panel = DatasetDownload(ctx)
        self.panel.changed.connect(self.changed)
        row.addWidget(self.combo)
        row.addWidget(self.panel.status)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.panel)

    def set_task(self, task: Task) -> None:
        current = self.card_id
        self.combo.blockSignals(True)
        self.combo.clear()
        for card in self.ctx.catalog.datasets.all():
            if task in card.tasks and card.adapter and card.maturity != "planned":
                self.combo.addItem(f"{card.name} · {card.size}", card.id)
        self.combo.setCurrentIndex(max(self.combo.findData(current), 0))
        self.combo.blockSignals(False)
        self.panel.set_card(self.card_id)

    def select(self, card_id: str) -> None:
        index = self.combo.findData(card_id)
        if index >= 0:
            self.combo.setCurrentIndex(index)

    @property
    def card_id(self) -> str | None:
        return self.combo.currentData()

    @property
    def ready(self) -> bool:
        return self.panel.ready


class ModelChooser(QWidget):
    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._asks = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.combo = _combo()
        self.combo.currentIndexChanged.connect(self._refresh)
        self.about = label("", "CardCaption", wrap=True)
        self.weights = _combo()
        self.weights.currentIndexChanged.connect(self._refresh_weights)
        self.weights_note = label("", "CardCaption", wrap=True)
        self.terms = label("", "Body", wrap=True)
        self.acknowledge = QCheckBox("I accept these terms")
        self.acknowledge.toggled.connect(self.changed)
        self.weights_title = label("Starting weights", "CardCaption")
        for widget in (self.combo, self.about, self.weights_title, self.weights,
                       self.weights_note, self.terms, self.acknowledge):
            layout.addWidget(widget)

    def set_task(self, task: Task) -> None:
        current = self.card_id
        self.combo.blockSignals(True)
        self.combo.clear()
        for card in self.ctx.catalog.models.all():
            if task in card.tasks and self.ctx.catalog.models.status(card.id) == "ready":
                self.combo.addItem(card.name, card.id)
        self.combo.setCurrentIndex(max(self.combo.findData(current), 0))
        self.combo.blockSignals(False)
        self._refresh()

    def select(self, card_id: str, pretrained: str | None = None) -> None:
        index = self.combo.findData(card_id)
        if index >= 0:
            self.combo.setCurrentIndex(index)
        self.weights.setCurrentIndex(max(self.weights.findData(pretrained), 0))

    @property
    def card_id(self) -> str | None:
        return self.combo.currentData()

    @property
    def pretrained(self) -> str | None:
        return self.weights.currentData()

    @property
    def acknowledged(self) -> bool:
        return not self._asks or self.acknowledge.isChecked()

    def _refresh(self) -> None:
        self.weights.blockSignals(True)
        self.weights.clear()
        self.weights.addItem("None: train from scratch", None)
        if self.card_id is not None:
            card = self.ctx.catalog.models.get(self.card_id)
            self.about.setText(f"{card.description} {card.architecture}. Code licence: {card.license.name}.")
            for weights in card.weights:
                self.weights.addItem(f"{weights.id} · {weights.license.label}", weights.id)
        self.weights.blockSignals(False)
        for widget in (self.weights_title, self.weights):  # only models with published weights
            widget.setVisible(self.weights.count() > 1)
        self._refresh_weights()

    def _refresh_weights(self) -> None:
        card_id, weights_id = self.card_id, self.pretrained
        if card_id is None or weights_id is None:
            self.weights_note.setText("")
            self._asks = False
            self.acknowledge.hide()
            self.terms.hide()
        else:
            weights = self.ctx.catalog.models.get(card_id).weights_by_id(weights_id)
            decision = download_policy(weights)
            self.weights_note.setText(f"{weights.description} Downloaded by the run into the workspace's "
                                      f"models/ folder. {weights.license.notes}".strip())
            self._asks = bool(decision.acknowledgement)
            self.acknowledge.setVisible(self._asks)
            self.terms.setVisible(self._asks)
            self.terms.setText(decision.acknowledgement or "")
        self.weights_note.setVisible(bool(self.weights_note.text()))
        self.changed.emit()
