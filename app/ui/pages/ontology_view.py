"""Ontology Explorer: the 1000 ImageNet-1k classes and their place in WordNet. Search a class
or any English noun, follow the path from "entity" down, step to narrower synsets, and see
which ImageNet classes fall under a synset. Class names only: never ImageNet images."""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from labs.ontology.imagenet import imagenet_classes, search_classes
from labs.ontology.wordnet import Node, WordNet

from ...services.context import AppContext
from ..theme.tokens import COLORS
from ..widgets import Card, DataTable, KeyValues, ResponsiveRow, label
from .builder.choosers import DatasetDownload

__all__ = ["OntologyView"]

WORDNET_CARD = "wordnet"


def _link(node: Node) -> str:
    return f'<a href="{node.wnid}" style="color:{COLORS["accent_hover"]}; text-decoration:none">' \
           f"{escape(node.lemma)}</a>"


def _links(nodes, limit: int = 60) -> str:  # noqa: ANN001 (Node or ImagenetClass)
    nodes = list(nodes)
    nodes = [n if isinstance(n, Node) else Node(n.wnid, n.wnid, n.name) for n in nodes]
    shown = ", ".join(_link(n) for n in nodes[:limit])
    return shown + (f" and {len(nodes) - limit} more" if len(nodes) > limit else "")


class OntologyView(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.wordnet: WordNet | None = None
        self._loading: str | None = None
        self.keys: list[str] = []
        self.current: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        intro = Card("Ontology Explorer", "ImageNet-1k classes in WordNet; class names only, never images")
        self.search = QLineEdit()
        self.search.setPlaceholderText("A class, a WordNet id (n02084071), a class number, or any noun…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _: self.refresh())
        intro.add(self.search)
        self.state = label("", "CardCaption", wrap=True)
        intro.add(self.state)
        self.download = DatasetDownload(ctx, show_status=False)
        intro.add(self.download)
        layout.addWidget(intro)

        results = Card("Matches", padded=False)
        self.table = DataTable(("Name", "WordNet id", "ImageNet"), mono_columns=(1, 2), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        results.add(self.table)
        self.detail = Card("")
        self.values = KeyValues((("WordNet id", ""), ("Synset", ""), ("ImageNet class", ""), ("Names", "")))
        self.gloss = label("", "Body", wrap=True)
        self.path = label("", "Body", wrap=True)
        self.narrower = label("", "Body", wrap=True)
        self.below = label("", "Body", wrap=True)
        for widget in (self.path, self.narrower, self.below):
            widget.setTextFormat(Qt.TextFormat.RichText)
            widget.setOpenExternalLinks(False)
            widget.linkActivated.connect(self.show_synset)
        for widget in (self.values, self.gloss, label("Path from the root", "CardCaption"), self.path,
                       label("Narrower synsets", "CardCaption"), self.narrower,
                       label("ImageNet classes at or below it", "CardCaption"), self.below):
            self.detail.add(widget)
        layout.addWidget(ResponsiveRow([(results, 2), (self.detail, 3)], breakpoint=900))

        ctx.downloads.finished.connect(self._downloaded)
        ctx.jobs.job_finished.connect(self._loaded)
        self._load_wordnet()
        self.refresh()

    # WordNet --------------------------------------------------------------------
    @property
    def wordnet_ready(self) -> bool:
        return WORDNET_CARD in self.ctx.catalog.datasets and \
            self.ctx.catalog.datasets.status(WORDNET_CARD) == "ready"

    def _load_wordnet(self) -> None:
        self.download.setVisible(not self.wordnet_ready and WORDNET_CARD in self.ctx.catalog.datasets)
        if self.wordnet_ready and self.wordnet is None and self._loading is None:
            root = self.ctx.paths.datasets

            def load(cancel, progress) -> WordNet:  # noqa: ANN001
                wordnet = WordNet(root)
                _ = wordnet._ancestors  # the one slow step (a few seconds), off the UI thread
                return wordnet

            self._loading = self.ctx.jobs.submit_task(load, title="Load WordNet", quiet=True)
        elif not self.wordnet_ready and WORDNET_CARD in self.ctx.catalog.datasets:
            self.download.set_card(WORDNET_CARD)
        self._state_text()

    def _state_text(self) -> None:
        if self.wordnet is not None:
            text = "WordNet 3.0 is loaded: paths, narrower synsets and any noun are available."
        elif self._loading is not None:
            text = "Loading WordNet…"
        else:
            text = ("Without WordNet you can search the ImageNet class list. Download WordNet 3.0 "
                    "(about 10 MB, permissive WordNet licence) to see the hierarchy.")
        self.state.setText(text)

    def _loaded(self, job_id: str, status: str) -> None:
        if job_id != self._loading:
            return
        self._loading = None
        job = self.ctx.jobs.job(job_id)
        if status == "completed":
            self.wordnet = job.result
        else:
            self.state.setText(f"WordNet could not be loaded: {job.error}")
            return
        self._state_text()
        self.refresh()

    def _downloaded(self, card_id: str, status: str, error: str) -> None:
        if card_id == WORDNET_CARD and status == "completed":
            self._load_wordnet()

    # Search and detail --------------------------------------------------------------
    def refresh(self) -> None:
        text = self.search.text()
        rows = [(c.name, c.wnid, str(c.index)) for c in search_classes(text)]
        if self.wordnet is not None and text.strip() and not text.strip().isdigit():
            known = {wnid for _, wnid, _ in rows}
            rows += [(node.name, node.wnid, "—") for node in self.wordnet.lookup(text)
                     if node.wnid not in known]
        self.keys = [wnid for _, wnid, _ in rows]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for row in rows:
            self.table.add_row(row)
        self.table.blockSignals(False)
        if self.keys:
            self.show_synset(self.current if self.current in self.keys else self.keys[0])
        self.detail.setVisible(bool(self.keys))

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_synset(self.keys[rows[0].row()])

    def show_synset(self, wnid: str) -> None:
        self.current = wnid
        if wnid in self.keys:
            self.table.blockSignals(True)
            self.table.selectRow(self.keys.index(wnid))
            self.table.blockSignals(False)
        item = next((c for c in imagenet_classes() if c.wnid == wnid), None)
        self.values.set_value("WordNet id", wnid)
        self.values.set_value("ImageNet class", f"{item.index}" if item else "not an ImageNet-1k class")
        if self.wordnet is None:
            self.detail.set_title(item.name if item else wnid)
            self.values.set_value("Synset", "—")
            self.values.set_value("Names", ", ".join(item.names) if item else "—")
            self.gloss.setText(item.gloss if item else "")
            for widget in (self.path, self.narrower, self.below):
                widget.setText("Download WordNet to see this.")
            return
        info = self.wordnet.info(wnid)
        self.detail.set_title(info.node.lemma)
        self.values.set_value("Synset", info.node.name)
        self.values.set_value("Names", ", ".join(info.lemmas))
        self.gloss.setText(info.definition + (f" — “{info.examples[0]}”" if info.examples else ""))
        self.path.setText("".join("<p style='margin:0 0 6px 0'>" + " › ".join(_link(n) for n in path) + "</p>"
                                  for path in info.paths[:3]))
        self.narrower.setText(_links(info.hyponyms) or "None: this is a leaf.")
        self.below.setText(f"{len(info.imagenet)}: " + _links(info.imagenet) if info.imagenet else "None.")
