"""NLP / LLM lab: chat with a small language model, look at tokenizers and embeddings, and train text
classifiers (TF-IDF in seconds, or a fine-tuned transformer) that are tracked like every run and
tested on Model Testing. Models come from the Hugging Face Hub with their licences checked first."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Modality, Task
from labs.nlp.presets import TEXT_PRESETS

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Page, Segmented
from ..lab_parts import LabDatasets, PresetGrid, PretrainedModels
from .tools import ChatTool, EmbeddingTool, TokenizerTool

__all__ = ["NlpLlmPage"]

VIEWS = (("classify", "Train classifiers"), ("chat", "Chat"), ("tokens", "Tokenizer"), ("embed",
                                                                                        "Embeddings"))


class NlpLlmPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("NLP / LLM", "text classifiers, a local chat model, tokenizers and embeddings",
                         parent)
        self.ctx = ctx
        testing = QPushButton(icon("test-tube"), "Model Testing")
        testing.clicked.connect(lambda: ctx.navigate("testing"))
        builder = QPushButton(icon("sliders-horizontal"), "Experiment Builder")
        builder.clicked.connect(lambda: ctx.navigate("experiment_builder"))
        self.head.add_action(testing)
        self.head.add_action(builder)
        self.switch = Segmented(VIEWS)
        self.body.addWidget(self.switch)

        self.presets = PresetGrid(ctx, TEXT_PRESETS,
                                  "TF-IDF runs take seconds on a CPU; fine-tuning wants a GPU")
        self.datasets = LabDatasets(ctx, lambda card: card.modality == Modality.TEXT and
                                    Task.TEXT_CLASSIFICATION in card.tasks)
        self.chat = ChatTool(ctx)
        self.tokens = TokenizerTool(ctx)
        self.embed = EmbeddingTool(ctx)
        self.models = PretrainedModels(ctx, {Task.TEXT_GENERATION, Task.TEXT_CLASSIFICATION},
                                       "chat models, encoders for embeddings and fine-tuning")
        self.models.changed.connect(self.refresh)
        self.groups: dict[str, QWidget] = {}
        for key, widgets in (("classify", (self.presets, self.datasets)), ("chat", (self.chat,)),
                             ("tokens", (self.tokens,)), ("embed", (self.embed,))):
            holder = QWidget()
            column = QVBoxLayout(holder)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(12)
            for widget in widgets:
                column.addWidget(widget)
            self.groups[key] = holder
            self.body.addWidget(holder)
        self.body.addWidget(self.models)
        self.body.addStretch(1)
        self.switch.changed.connect(self.show_view)
        ctx.downloads.finished.connect(lambda *_: self.presets.refresh())
        self.refresh()
        self.show_view("classify")

    def show_view(self, key: str) -> None:
        if self.switch.current != key:
            self.switch.buttons[key].setChecked(True)
        for name, holder in self.groups.items():
            holder.setVisible(name == key)

    def refresh(self) -> None:
        self.presets.refresh()
        for chooser in (self.chat.model, self.tokens.model, self.embed.model):
            chooser.refresh()
        self.chat.refresh_buttons()
