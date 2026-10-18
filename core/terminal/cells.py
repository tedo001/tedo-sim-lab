"""Cells and their attributes, shared by the screen, the parser and the widget that paints them.

A cell is ``(text, attr)``: one character (with any combining marks), or ``""`` for the right
half of a wide character; ``attr`` is ``(fg, bg, flags)``. Colours are ``DEFAULT``, a palette
index 0-255, or ``RGB | 0xRRGGBB``.
"""

from __future__ import annotations

import unicodedata

__all__ = ["BLINK", "BOLD", "DEC_GRAPHICS", "DEFAULT", "DIM", "HIDDEN", "ITALIC", "PLAIN",
           "REVERSE", "RGB", "Row", "STRIKE", "UNDERLINE", "char_width"]

BOLD, DIM, ITALIC, UNDERLINE, BLINK, REVERSE, HIDDEN, STRIKE = (1 << i for i in range(8))
DEFAULT = -1
RGB = 1 << 24
PLAIN = (DEFAULT, DEFAULT, 0)

#: DEC special graphics (``ESC ( 0``): the line-drawing set older curses programs use.
DEC_GRAPHICS = str.maketrans(dict(zip(
    "`abcdefghijklmnopqrstuvwxyz{|}~",
    "◆▒␉␌␍␊°±␤␋┘┐┌└┼⎺⎻─⎼⎽├┤┴┬│≤≥π≠£·", strict=True)))


def char_width(char: str) -> int:
    """Cells ``char`` takes: 2 for wide (CJK, most emoji), 0 for combining marks, else 1."""
    if char.isascii():
        return 1
    if unicodedata.combining(char) or char in "\u200b\u200c\u200d\ufe0e\ufe0f":
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


class Row(list):
    """One line of cells; ``wrapped`` = it continues on the next line (long text, not Enter)."""

    __slots__ = ("wrapped",)

    def __init__(self, cells=()) -> None:
        super().__init__(cells)
        self.wrapped = False
