"""Painting a terminal screen: the 256-colour palette (and true colour), cell attributes (bold,
dim, italic, underline, reverse, hidden, strike-through), the selection and the cursor, on the
grid of a monospaced font. Runs of plain ASCII are drawn in one call; anything else per cell, so
glyphs from fallback fonts cannot push the grid out of line."""

from __future__ import annotations

from math import ceil

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen

from core.terminal.cells import BOLD, DEFAULT, DIM, HIDDEN, ITALIC, REVERSE, RGB, STRIKE, UNDERLINE

from ..theme import mono_font
from ..theme.tokens import COLORS

__all__ = ["CellFonts", "TerminalPalette", "paint_cursor", "paint_rows"]


class TerminalPalette:
    def __init__(self) -> None:
        self.fg = QColor(COLORS["term_fg"])
        self.bg = QColor(COLORS["term_bg"])
        self.cursor = QColor(COLORS["term_cursor"])
        self.selection = QColor(COLORS["term_selection"])
        base = [QColor(COLORS[f"term_{i}"]) for i in range(16)]
        cube = [QColor(r and r * 40 + 55, g and g * 40 + 55, b and b * 40 + 55)
                for r in range(6) for g in range(6) for b in range(6)]
        greys = [QColor(8 + 10 * i, 8 + 10 * i, 8 + 10 * i) for i in range(24)]
        self.colours = base + cube + greys
        self._cache: dict = {}

    def colour(self, value: int, default: QColor) -> QColor:
        if value == DEFAULT:
            return default
        if value & RGB:
            return QColor((value >> 16) & 255, (value >> 8) & 255, value & 255)
        return self.colours[value] if 0 <= value < 256 else default

    def resolve(self, attr: tuple[int, int, int], selected: bool, reverse_screen: bool):
        """(text colour, background to fill or None, flags) for a cell."""
        key = (attr, selected, reverse_screen)
        found = self._cache.get(key)
        if found is not None:
            return found
        fg, bg, flags = attr
        if flags & BOLD and 0 <= fg < 8:
            fg += 8  # bold shows the bright colour, as xterm does
        default_fg, default_bg = (self.bg, self.fg) if reverse_screen else (self.fg, self.bg)
        text, back = self.colour(fg, default_fg), self.colour(bg, default_bg)
        if flags & REVERSE:
            text, back = back, text
        if flags & DIM:
            text = QColor((text.red() + back.red()) // 2, (text.green() + back.green()) // 2,
                          (text.blue() + back.blue()) // 2)
        if flags & HIDDEN:
            text = back
        if selected:
            back = self.selection
        found = (text, None if back.rgb() == self.bg.rgb() else back, flags)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = found
        return found


class CellFonts:
    """The regular, bold and italic faces at one size, and the cell grid they define."""

    def __init__(self, pixel_size: int) -> None:
        self.pixel_size = pixel_size
        regular = mono_font(pixel_size)
        regular.setKerning(False)
        bold = QFont(regular)
        bold.setWeight(QFont.Weight.DemiBold)
        italic = QFont(regular)
        italic.setItalic(True)
        bold_italic = QFont(bold)
        bold_italic.setItalic(True)
        self.faces = {0: regular, BOLD: bold, ITALIC: italic, BOLD | ITALIC: bold_italic}
        metrics = QFontMetricsF(regular)
        self.width = metrics.horizontalAdvance("M")
        self.height = float(ceil(metrics.lineSpacing()))
        self.ascent = metrics.ascent() + (self.height - metrics.lineSpacing()) / 2

    def face(self, flags: int) -> QFont:
        return self.faces[flags & (BOLD | ITALIC)]


def paint_rows(painter: QPainter, rows: list, fonts: CellFonts, palette: TerminalPalette, *,
               first_line: int, selection: tuple[tuple[int, int], tuple[int, int]] | None,
               reverse_screen: bool) -> None:
    """Draw ``rows`` (line numbers from ``first_line``); ``selection`` = ((line, col), (line,
    col)) with the end exclusive."""
    width, height, ascent = fonts.width, fonts.height, fonts.ascent
    for y, row in enumerate(rows):
        if row is None:
            continue
        line, top = first_line + y, y * height
        cells = [(cell, palette.resolve(cell[1], _selected(selection, line, x), reverse_screen))
                 for x, cell in enumerate(row)]
        # Backgrounds, in runs of one colour.
        x = 0
        while x < len(cells):
            back = cells[x][1][1]
            end = x + 1
            while end < len(cells) and cells[end][1][1] == back:
                end += 1
            if back is not None:
                painter.fillRect(QRectF(x * width, top, (end - x) * width, height), back)
            x = end
        # Text.
        x = 0
        while x < len(cells):
            (char, _attr), (text_colour, _back, flags) = cells[x]
            if char in ("", " ") and not flags & (UNDERLINE | STRIKE):
                x += 1
                continue
            end = x + 1
            if char.isascii() and len(char) == 1:
                while (end < len(cells) and cells[end][1] == cells[x][1] and cells[end][0][0].isascii()
                       and len(cells[end][0][0]) == 1):
                    end += 1
                text = "".join(cell[0][0] for cell in cells[x:end])
            else:
                text = char
            painter.setFont(fonts.face(flags))
            painter.setPen(text_colour)
            if text.strip():
                painter.drawText(QPointF(x * width, top + ascent), text)
            wide = end == x + 1 and end < len(cells) and cells[end][0][0] == ""
            span = (2 if wide else end - x) * width
            if flags & UNDERLINE:
                under = top + ascent + 2
                painter.drawLine(QPointF(x * width, under), QPointF(x * width + span, under))
            if flags & STRIKE:
                middle = top + height / 2
                painter.drawLine(QPointF(x * width, middle), QPointF(x * width + span, middle))
            x = end


def paint_cursor(painter: QPainter, fonts: CellFonts, palette: TerminalPalette, *, x: int, y: int,
                 cell: tuple[str, tuple[int, int, int]], shape: int, focused: bool) -> None:
    """Block (inverted character), underline or bar; an outline when the terminal has no focus."""
    rect = QRectF(x * fonts.width, y * fonts.height, fonts.width, fonts.height)
    if not focused:
        painter.setPen(QPen(palette.cursor, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        return
    if shape in (3, 4):
        painter.fillRect(QRectF(rect.left(), rect.bottom() - 2, rect.width(), 2), palette.cursor)
    elif shape in (5, 6):
        painter.fillRect(QRectF(rect.left(), rect.top(), 2, rect.height()), palette.cursor)
    else:
        painter.fillRect(rect, palette.cursor)
        char, attr = cell
        if char.strip():
            painter.setFont(fonts.face(attr[2]))
            painter.setPen(palette.bg)
            painter.drawText(QPointF(rect.left(), rect.top() + fonts.ascent), char)


def _selected(selection, line: int, col: int) -> bool:
    if selection is None:
        return False
    (start_line, start_col), (end_line, end_col) = selection
    return (start_line, start_col) <= (line, col) < (end_line, end_col)
