"""The lab's Markdown documentation, rendered in place."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QSizePolicy, QWidget

from ...services.context import AppContext
from ..widgets import Card, Page, label
from ..widgets.markdown import MarkdownView

__all__ = ["DocumentationPage", "find_documents"]

#: Shown first, in this order, when present; anything else under docs/ follows.
_PREFERRED = ("README.md", "CLAUDE.md", "ROADMAP.md")


def find_documents(code_root: Path) -> list[Path]:
    found = [code_root / name for name in _PREFERRED if (code_root / name).is_file()]
    docs = code_root / "docs"
    if docs.is_dir():
        found += sorted(docs.rglob("*.md"))
    return found


class DocumentationPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Documentation", str(ctx.paths.code_root), parent)
        self.documents = find_documents(ctx.paths.code_root)

        card = Card(padded=False)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        row = QHBoxLayout()
        row.setSpacing(0)
        self.index = QListWidget()
        self.index.setObjectName("DocList")
        self.index.setFixedWidth(200)
        self.view = MarkdownView()
        self.view.setMinimumHeight(480)
        for document in self.documents:
            item = QListWidgetItem(str(document.relative_to(ctx.paths.code_root)))
            item.setData(Qt.ItemDataRole.UserRole, str(document))
            self.index.addItem(item)
        self.index.currentItemChanged.connect(self._show)
        row.addWidget(self.index)
        row.addWidget(self.view, 1)
        card.add(row)
        self.body.addWidget(card, 1)

        if self.documents:
            self.index.setCurrentRow(0)
        else:
            card.hide()
            self.body.insertWidget(0, label("No Markdown documents were found next to the code.",
                                            "Body"))
            self.body.addStretch(1)

    def _show(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            text = f"Could not read {path.name}: {exc.strerror}"
        self.view.set_markdown(text, base=path.parent)
