"""Markdown rendered with the lab's typography.

Qt's Markdown importer builds text formats directly and ignores style sheets,
so headings, paragraph spacing and code are adjusted block by block after the
document is loaded.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import QTextBrowser, QWidget

from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY

__all__ = ["MarkdownView", "style_markdown"]

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
            fmt.setLineHeight(145, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor.mergeCharFormat(_char(FONT_FAMILY, colour=COLORS["text_dim"]))
            _style_inline_code(block, cursor)
        cursor.setBlockFormat(fmt)
        block = block.next()
    cursor.endEditBlock()


class MarkdownView(QTextBrowser):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DocView")
        self.setOpenExternalLinks(True)

    def set_markdown(self, text: str) -> None:
        self.setMarkdown(text)
        style_markdown(self.document())
