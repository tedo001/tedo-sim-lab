"""Phase 10: the search index and documents with pictures."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage, QTextDocument

from app.ui.pages.documentation import find_documents
from app.ui.shell.search_index import search_entries
from app.ui.widgets.markdown import MarkdownView
from core.common.paths import CODE_ROOT


def test_search_entries_cover_every_kind(ctx) -> None:
    keys = {key for _label, key in search_entries(ctx)}
    assert {"page:home", "dataset:mnist", "model:layer_stack"} <= keys
    assert len([k for k in keys if k.startswith("model:")]) == len(ctx.catalog.models)


def test_markdown_images_are_found_and_fitted(qtbot, tmp_path) -> None:
    QImage(1600, 800, QImage.Format.Format_RGB32).save(str(tmp_path / "wide.png"))
    view = MarkdownView()
    qtbot.addWidget(view)
    view.set_markdown("# Title\n\n![wide](wide.png)\n\n| a |\n| --- |\n| ![w](wide.png) |\n\n![gone](no.png)",
                      base=tmp_path)
    document = view.document()
    first = document.resource(QTextDocument.ResourceType.ImageResource, QUrl("tedo-image-0.png"))
    in_table = document.resource(QTextDocument.ResourceType.ImageResource, QUrl("tedo-image-1.png"))
    assert first.width() == MarkdownView.IMAGE_WIDTH and first.height() == MarkdownView.IMAGE_WIDTH // 2
    assert in_table.width() == MarkdownView.IMAGE_WIDTH // 2
    assert "[image: gone]" in view.toPlainText()


def test_documentation_lists_the_readme_first() -> None:
    names = [path.name for path in find_documents(CODE_ROOT)]
    assert names[:3] == ["README.md", "CLAUDE.md", "ROADMAP.md"]
    readme = (CODE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "ROADMAP.md" in readme and "CLAUDE.md" in readme


def test_the_readme_lists_every_page() -> None:
    """Every page in the sidebar is in the README, and every image it shows exists: the owner's rule is
    that the README is updated with every change."""
    import re

    from app.navigation import NAV

    readme = (CODE_ROOT / "README.md").read_text(encoding="utf-8")
    shown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert all((CODE_ROOT / name).is_file() for name in shown)
    missing = [spec.id for spec in NAV if spec.in_sidebar and f"| {spec.title} |" not in readme]
    assert not missing, f"the README does not list {missing}"
