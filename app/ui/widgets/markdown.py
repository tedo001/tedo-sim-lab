"""Markdown rendered with the lab's typography.

Qt's Markdown importer builds text formats directly and ignores style sheets,
so headings, paragraph spacing and code are adjusted block by block after the
document is loaded.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QFont, QImage, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import QTextBrowser, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["MarkdownView", "style_markdown"]

_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

_HEADING_PX = {1: 22, 2: 17, 3: 14, 4: 13, 5: 13, 6: 13}


def _char(family: str, pixel_size: int | None = None, weight: int | None = None,
          colour: str | None = None) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setFontFamilies([family])
    if pixel_size:
        font = QFont(family)
        font.setPixelSize(pixel_size)
        fmt.setFont(font, QTextCharFormat.FontPropertiesInheritanceBehavior.FontPropertiesSpecifiedOnly)
    if weight:
        fmt.setFontWeight(weight)
    if colour:
        fmt.setForeground(QColor(colour))
    return fmt


def _style_inline_code(block, cursor: QTextCursor) -> None:
    it = block.begin()
    while not it.atEnd():
        fragment = it.fragment()
        if fragment.isValid() and fragment.charFormat().fontFixedPitch():
            cursor.setPosition(fragment.position())
            cursor.setPosition(fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(_char(MONO_FAMILY, 12, colour=COLORS["text"]))
        it += 1


def _has_image(block) -> bool:
    iterator = block.begin()
    while not iterator.atEnd():
        if iterator.fragment().charFormat().isImageFormat():
            return True
        iterator += 1
    return False


def style_markdown(document: QTextDocument) -> None:
    """Apply heading sizes, spacing and the monospace face to a Markdown document."""
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    block = document.begin()
    while block.isValid():
        fmt = QTextBlockFormat(block.blockFormat())
        cursor.setPosition(block.position())
        cursor.setPosition(block.position() + max(block.length() - 1, 0),
                           QTextCursor.MoveMode.KeepAnchor)
        level = fmt.headingLevel()
        if level:
            fmt.setTopMargin(20 if level <= 2 else 14)
            fmt.setBottomMargin(6)
            cursor.mergeCharFormat(_char(FONT_FAMILY, _HEADING_PX[level], 600, COLORS["text"]))
        elif fmt.nonBreakableLines():  # fenced or indented code; space only around the whole block
            fmt.setBackground(QColor(COLORS["raised"]))
            fmt.setTopMargin(0 if block.previous().blockFormat().nonBreakableLines() else 4)
            fmt.setBottomMargin(0 if block.next().blockFormat().nonBreakableLines() else 12)
            fmt.setLeftMargin(0)
            cursor.mergeCharFormat(_char(MONO_FAMILY, 12, colour=COLORS["text"]))
        else:
            fmt.setBottomMargin(8)
            if not _has_image(block):  # a picture's "line" must not grow by half its height
                fmt.setLineHeight(145, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor.mergeCharFormat(_char(FONT_FAMILY, colour=COLORS["text_dim"]))
            _style_inline_code(block, cursor)
        cursor.setBlockFormat(fmt)
        block = block.next()
    cursor.endEditBlock()


class MarkdownView(QTextBrowser):
    IMAGE_WIDTH = 720

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DocView")
        self.setOpenExternalLinks(True)

    def set_markdown(self, text: str, base: Path | None = None) -> None:
        """Show ``text``. Images are read relative to ``base`` and shrunk to fit: at most
        ``IMAGE_WIDTH`` wide, half that in a table row."""
        images: dict[str, QImage] = {}

        def swap(match: re.Match[str], in_table: bool) -> str:
            source = Path(match.group(2))
            path = source if source.is_absolute() or base is None else base / source
            image = QImage(str(path))
            if image.isNull():
                return f"*[image: {match.group(1) or source.name}]*"
            limit = self.IMAGE_WIDTH // 2 if in_table else self.IMAGE_WIDTH
            if image.width() > limit:
                image = image.scaledToWidth(limit, Qt.TransformationMode.SmoothTransformation)
            name = f"tedo-image-{len(images)}.png"
            images[name] = image
            return f"![{match.group(1)}]({name})"

        lines = [_IMAGE.sub(lambda m, row=line.lstrip().startswith("|"): swap(m, row), line)
                 if "![" in line else line for line in text.splitlines()]
        self.setMarkdown("\n".join(lines))
        for name, image in images.items():  # (loading the text resets resources: add them after)
            self.document().addResource(QTextDocument.ResourceType.ImageResource, QUrl(name), image)
        style_markdown(self.document())
        self.document().markContentsDirty(0, self.document().characterCount())
