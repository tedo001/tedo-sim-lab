"""The NLP / LLM lab's tools, each answered by the inference worker with a downloaded model:
chat with a small language model (streamed token by token), see how a tokenizer splits a text,
and compare texts by their embeddings (similarity to a query, and a 3-D map of the texts)."""

from __future__ import annotations

import html
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QWidget,
)

from core.common.vocab import Task
from labs.common.explore import principal_components

from ....services.context import AppContext
from ...icons import icon
from ...theme.tokens import COLORS
from ...widgets import Card, label
from ...widgets.plots import BarList
from ...widgets.scatter3d import ScatterView
from ..lab_parts import DownloadedModels

__all__ = ["ChatTool", "EmbeddingTool", "TokenizerTool"]


def _row(*widgets: QWidget, stretch: int | None = None) -> QHBoxLayout:
    row = QHBoxLayout()
    for index, widget in enumerate(widgets):
        row.addWidget(widget, 1 if index == stretch else 0)
    if stretch is None:
        row.addStretch(1)
    return row


class ChatTool(Card):
    """A conversation with a downloaded instruction-tuned model; the reply streams in."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Chat", "a small language model on this computer; nothing leaves it")
        self.ctx = ctx
        self.messages: list[dict[str, str]] = []
        self.pending: str | None = None
        self.model = DownloadedModels(ctx, {Task.TEXT_GENERATION})
        self.model.currentIndexChanged.connect(lambda _: self.refresh_buttons())
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        self.temperature.setPrefix("temperature ")
        self.tokens = QSpinBox()
        self.tokens.setRange(16, 4096)
        self.tokens.setValue(256)
        self.tokens.setPrefix("up to ")
        self.tokens.setSuffix(" tokens")
        self.add(_row(self.model, self.temperature, self.tokens))
        self.system = QLineEdit("You are a helpful, concise assistant.")
        self.system.setPlaceholderText("System prompt (how the model should behave)")
        self.add(self.system)
        self.transcript = QTextBrowser()
        self.transcript.setObjectName("DocView")
        self.transcript.setMinimumHeight(220)
        self.add(self.transcript)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Ask something…")
        self.prompt.setFixedHeight(70)
        self.add(self.prompt)
        self.send_button = QPushButton(icon("send"), "Send")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self.send)
        self.stop_button = QPushButton(icon("circle-stop"), "Stop")
        self.stop_button.clicked.connect(lambda: self.pending and ctx.inference.cancel(self.pending))
        self.clear_button = QPushButton(icon("trash-2"), "New chat")
        self.clear_button.clicked.connect(self.clear)
        self.status = label("", "Mono")
        self.add(_row(self.send_button, self.stop_button, self.clear_button, self.status, stretch=3))
        self._draw("")
        self.refresh_buttons()

    def clear(self) -> None:
        self.messages = []
        self._draw("")

    def _draw(self, streaming: str) -> None:
        parts = []
        for message in self.messages + ([{"role": "assistant", "content": streaming}] if streaming else []):
            who = "You" if message["role"] == "user" else "Model"
            colour = COLORS["text_dim"] if message["role"] == "user" else COLORS["accent"]
            text = html.escape(message["content"]).replace("\n", "<br>")
            parts.append(f'<p><b style="color:{colour}">{who}</b><br>{text}</p>')
        self.transcript.setHtml("".join(parts) or f'<p style="color:{COLORS["text_faint"]}">'
                                "Download a chat model below, then ask something.</p>")
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def refresh_buttons(self) -> None:
        self.send_button.setEnabled(self.model.repo is not None and self.pending is None)
        self.stop_button.setEnabled(self.pending is not None)

    def send(self) -> None:
        text = self.prompt.toPlainText().strip()
        if not text or self.model.repo is None or self.pending:
            return
        self.prompt.clear()
        self.messages.append({"role": "user", "content": text})
        system = self.system.text().strip()
        messages = ([{"role": "system", "content": system}] if system else []) + self.messages
        streamed: list[str] = []

        def partial(data: dict[str, Any]) -> None:
            streamed.append(data.get("text", ""))
            self._draw("".join(streamed))

        def done(result: dict[str, Any]) -> None:
            self.pending = None
            self.messages.append({"role": "assistant", "content": result["text"]})
            self._draw("")
            self.status.setText(f"{result['tokens']} tokens · {result['tokens_per_s']:.1f}/s on "
                                f"{result['device']}"
                                + (" · stopped" if result.get("stopped") else ""))
            self.refresh_buttons()

        def failed(message: str) -> None:
            self.pending = None
            self.status.setText(f"Failed: {message}")
            self.refresh_buttons()

        self.status.setText("Thinking…")
        self.pending = self.ctx.inference.request(
            "generate", {"model": self.model.repo, "messages": messages,
                         "temperature": self.temperature.value(),
                         "max_new_tokens": self.tokens.value()},
            on_partial=partial, on_done=done, on_error=failed)
        self.refresh_buttons()


class TokenizerTool(Card):
    """How a model's tokenizer splits a text: each token with its id, alternately shaded."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Tokenizer", "how a model reads text: pieces of words, each a number")
        self.ctx = ctx
        self.model = DownloadedModels(ctx, {Task.TEXT_GENERATION, Task.TEXT_CLASSIFICATION})
        go = QPushButton(icon("play"), "Split")
        go.clicked.connect(self.split)
        self.add(_row(self.model, go))
        self.text = QPlainTextEdit("Tokenizers split unfamiliar words like antidisestablishmentarianism into "
                                   "pieces.")
        self.text.setFixedHeight(70)
        self.add(self.text)
        self.view = QTextBrowser()
        self.view.setObjectName("DocView")
        self.view.setMinimumHeight(120)
        self.add(self.view)
        self.summary = label("", "Mono")
        self.add(self.summary)

    def split(self) -> None:
        if self.model.repo is None:
            self.summary.setText("Download a model below first.")
            return

        def done(result: dict[str, Any]) -> None:
            shades = (COLORS["selected"], COLORS["raised"])
            chips = [f'<span style="background:{shades[i % 2]}; color:{COLORS["text"]}">'
                     f'{html.escape(token).replace(" ", "·")}</span><sub '
                     f'style="color:{COLORS["text_faint"]}">'
                     f'{token_id}</sub> ' for i, (token, token_id) in enumerate(zip(result["tokens"],
                                                                                    result["ids"],
                                                                                     strict=True))]
            self.view.setHtml(f'<p style="font-family:JetBrains Mono; line-height:160%">{"".join(chips)}</p>')
            self.summary.setText(f"{len(result['ids'])} tokens for {result['characters']} characters · "
                                 f"vocabulary {result['vocabulary']:,} · {result['kind']}")

        self.summary.setText("Working…")
        self.ctx.inference.request("tokenize", {"model": self.model.repo, "text": self.text.toPlainText()},
                                   on_done=done, on_error=lambda m: self.summary.setText(f"Failed: {m}"))


class EmbeddingTool(Card):
    """Texts as vectors: similarity to a query, and the texts as points in 3-D (principal components)."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Embeddings", "meaning as a direction: similar texts point the same way")
        self.ctx = ctx
        self.model = DownloadedModels(ctx, {Task.TEXT_CLASSIFICATION})
        go = QPushButton(icon("play"), "Compare")
        go.clicked.connect(self.compare)
        self.add(_row(self.model, go))
        self.texts = QPlainTextEdit("The cat sat on the mat.\nA kitten is sleeping on the rug.\nStocks fell "
                                    "sharply today.\nThe market lost value this morning.\nI love hiking in "
                                    "the mountains.")
        self.texts.setFixedHeight(110)
        self.add(label("One text per line", "CardCaption"))
        self.add(self.texts)
        self.query = QLineEdit("a cat resting indoors")
        self.query.setPlaceholderText("A query to rank the texts by")
        self.add(self.query)
        self.ranking = BarList()
        self.add(self.ranking)
        self.map = ScatterView("texts by their embeddings (first three principal components)")
        self.map.setMinimumHeight(280)
        self.add(self.map)
        self.summary = label("", "Mono")
        self.add(self.summary)

    def compare(self) -> None:
        texts = [line.strip() for line in self.texts.toPlainText().splitlines() if line.strip()]
        if self.model.repo is None or len(texts) < 2:
            self.summary.setText("Download an encoder below and give at least two texts.")
            return

        def done(result: dict[str, Any]) -> None:
            vectors = np.asarray(result["vectors"], dtype=float)
            if "similarity" in result:
                order = np.argsort(-np.asarray(result["similarity"]))
                self.ranking.set_rows([[texts[i], result["similarity"][i]] for i in order])
            scores, _share = principal_components(vectors, 3, standardize=False)
            points = np.zeros((len(texts), 3))
            points[:, :scores.shape[1]] = scores
            similarity = result.get("similarity")
            self.map.set_points(points, values=similarity or list(range(1, len(texts) + 1)),
                                value_label="similarity to the query" if similarity else "line",
                                axes=("PC 1", "PC 2", "PC 3"))
            self.summary.setText(f"{len(texts)} texts · {result['dimensions']} dimensions each")

        self.summary.setText("Working…")
        self.ctx.inference.request("embed", {"model": self.model.repo, "texts": texts,
                                             "query": self.query.text().strip() or None},
                                   on_done=done, on_error=lambda m: self.summary.setText(f"Failed: {m}"))
