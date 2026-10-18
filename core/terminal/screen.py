"""The screen of a terminal: a grid of character cells that a program draws on with VT100/xterm
control sequences (cursor moves, erases, scroll regions, colours, the alternate screen that vim
and nvim use), plus the scrollback above it. :class:`~core.terminal.parser.Parser` turns a
program's output into calls on this class; the app's terminal widget paints ``lines``.
Cells and their attributes are in :mod:`core.terminal.cells`.
"""

from __future__ import annotations

from collections import deque

from .cells import (
    BLINK,
    BOLD,
    DEC_GRAPHICS,
    DEFAULT,
    DIM,
    HIDDEN,
    ITALIC,
    PLAIN,
    REVERSE,
    RGB,
    STRIKE,
    UNDERLINE,
    Row,
    char_width,
)
from .editing import EditingMixin
from .modes import ModesMixin

__all__ = ["BLINK", "BOLD", "DEFAULT", "DIM", "HIDDEN", "ITALIC", "PLAIN", "REVERSE", "RGB",
           "Row", "STRIKE", "Screen", "UNDERLINE", "char_width"]

class Screen(EditingMixin, ModesMixin):
    def __init__(self, cols: int = 80, rows: int = 24, scrollback: int = 5000) -> None:
        self.cols, self.rows = max(2, cols), max(2, rows)
        self.scrollback: deque[Row] = deque(maxlen=max(0, scrollback))
        #: Lines that ever scrolled into the scrollback: a stable numbering for selections.
        self.history_total = 0
        #: Replies to queries (cursor position, device attributes), to write back to the program.
        self.responses: list[str] = []
        self.title = ""
        self.bells = 0
        #: Colours reported to programs that ask (OSC 10/11/12), as ``rrggbb``.
        self.reported_colours = {10: "e3e6ea", 11: "0f1114", 12: "e3e6ea"}
        self.reset()

    # State -------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Full reset (RIS): blank screen, default modes, cursor home; scrollback kept."""
        self.attr = PLAIN
        self.main = [self._blank_row() for _ in range(self.rows)]
        self.alt: list[Row] | None = None
        self.lines = self.main
        self.cx = self.cy = 0
        self.wrap_next = False
        self.top, self.bottom = 0, self.rows - 1
        self.reset_modes()
        self.tabs = set(range(8, self.cols, 8))
        self.saved: dict[bool, tuple] = {}
        self.charsets = ["B", "B"]
        self.shift = 0
        self.last_char = " "

    @property
    def in_alt_screen(self) -> bool:
        return self.alt is not None and self.lines is self.alt

    def _blank(self) -> tuple[str, tuple[int, int, int]]:
        # Erased cells take the current background (xterm's background colour erase).
        return (" ", (DEFAULT, self.attr[1], 0))

    def _blank_row(self, blank=None) -> Row:
        return Row([blank or (" ", PLAIN)] * self.cols)

    # Text ----------------------------------------------------------------------------------------
    def draw(self, text: str) -> None:
        """Print ``text`` at the cursor, wrapping at the right margin when autowrap is on."""
        if self.charsets[self.shift] == "0":
            text = text.translate(DEC_GRAPHICS)
        if text.isascii() and not self.insert_mode:
            self._draw_ascii(text)
        else:
            for char in text:
                self._draw_char(char)
        if text:
            self.last_char = text[-1]

    def _draw_ascii(self, text: str) -> None:
        attr, cols = self.attr, self.cols
        while text:
            if self.wrap_next and self.autowrap:
                self._wrap()
            row = self.lines[self.cy]
            space = cols - self.cx
            if not self.autowrap and len(text) > space:
                text = text[:space - 1] + text[-1]  # the overflow overwrites the last column
            chunk, text = text[:space], text[space:]
            row[self.cx:self.cx + len(chunk)] = [(c, attr) for c in chunk]
            self.cx += len(chunk)
            if self.cx >= cols:
                self.cx = cols - 1
                self.wrap_next = self.autowrap

    def _draw_char(self, char: str) -> None:
        width = char_width(char)
        if width == 0:  # a combining mark joins the character before it
            x = self.cx if self.wrap_next else self.cx - 1
            row = self.lines[self.cy]
            if 0 <= x < self.cols:
                if row[x][0] == "" and x > 0:
                    x -= 1
                row[x] = (row[x][0] + char, row[x][1])
            return
        if self.wrap_next and self.autowrap:
            self._wrap()
        if width == 2 and self.cx == self.cols - 1:
            if not self.autowrap:
                return
            self.lines[self.cy][self.cx] = self._blank()
            self._wrap()
        row = self.lines[self.cy]
        if self.insert_mode:
            row[self.cx:self.cx] = [self._blank()] * width
            del row[self.cols:]
        x = self.cx
        if row[x][0] == "" and x > 0:  # overwriting the right half of a wide character
            row[x - 1] = self._blank()
        if x + width < self.cols and row[x + width][0] == "":
            row[x + width] = self._blank()
        row[x] = (char, self.attr)
        if width == 2:
            row[x + 1] = ("", self.attr)
        self.cx += width
        if self.cx >= self.cols:
            self.cx = self.cols - 1
            self.wrap_next = self.autowrap

    def _wrap(self) -> None:
        self.lines[self.cy].wrapped = True
        self.cx = 0
        self.wrap_next = False
        self.index()

    def repeat_last(self, count: int) -> None:
        self.draw(self.last_char * max(1, count))

    # Control characters --------------------------------------------------------------------------
    def carriage_return(self) -> None:
        self.cx = 0
        self.wrap_next = False

    def linefeed(self) -> None:
        self.index()
        if self.newline_mode:
            self.carriage_return()

    def backspace(self) -> None:
        self.cx = max(0, self.cx - 1)
        self.wrap_next = False

    def tab(self, count: int = 1) -> None:
        for _ in range(max(1, count)):
            stops = [t for t in self.tabs if t > self.cx]
            self.cx = min(stops) if stops else self.cols - 1
        self.wrap_next = False

    def back_tab(self, count: int = 1) -> None:
        for _ in range(max(1, count)):
            stops = [t for t in self.tabs if t < self.cx]
            self.cx = max(stops) if stops else 0
        self.wrap_next = False

    def bell(self) -> None:
        self.bells += 1

    def set_tab_stop(self) -> None:
        self.tabs.add(self.cx)

    def clear_tab_stop(self, mode: int = 0) -> None:
        if mode == 0:
            self.tabs.discard(self.cx)
        elif mode == 3:
            self.tabs.clear()

    # Scrolling -----------------------------------------------------------------------------------
    def index(self) -> None:
        """Down one line; at the bottom of the scroll region the region scrolls up."""
        if self.cy == self.bottom:
            self.scroll_up(1)
        elif self.cy < self.rows - 1:
            self.cy += 1

    def reverse_index(self) -> None:
        if self.cy == self.top:
            self.scroll_down(1)
        elif self.cy > 0:
            self.cy -= 1

    def next_line(self) -> None:
        self.index()
        self.carriage_return()

    def scroll_up(self, count: int = 1) -> None:
        count = min(max(1, count), self.bottom - self.top + 1)
        full = self.top == 0 and self.bottom == self.rows - 1
        for _ in range(count):
            gone = self.lines.pop(self.top)
            if full and not self.in_alt_screen and self.scrollback.maxlen:
                self.scrollback.append(gone)
                self.history_total += 1
            self.lines.insert(self.bottom, self._blank_row(self._blank()))

    def scroll_down(self, count: int = 1) -> None:
        count = min(max(1, count), self.bottom - self.top + 1)
        for _ in range(count):
            del self.lines[self.bottom]
            self.lines.insert(self.top, self._blank_row(self._blank()))

    def set_scroll_region(self, top: int = 0, bottom: int = 0) -> None:
        """DECSTBM with 1-based ``top``/``bottom`` (0 = the edge); homes the cursor."""
        top = max(1, top or 1) - 1
        bottom = min(self.rows, bottom or self.rows) - 1
        if top < bottom:
            self.top, self.bottom = top, bottom
            self.cursor_position(1, 1)

    # Cursor ----------------------------------------------------------------------------------------
    def _limits(self) -> tuple[int, int]:
        return (self.top, self.bottom) if self.origin_mode else (0, self.rows - 1)

    def cursor_position(self, row: int = 1, col: int = 1) -> None:
        low, high = self._limits()
        self.cy = min(max(low, low + max(1, row) - 1), high)
        self.cx = min(max(0, max(1, col) - 1), self.cols - 1)
        self.wrap_next = False

    def cursor_up(self, count: int = 1) -> None:
        low = self.top if self.cy >= self.top else 0
        self.cy = max(low, self.cy - max(1, count))
        self.wrap_next = False

    def cursor_down(self, count: int = 1) -> None:
        high = self.bottom if self.cy <= self.bottom else self.rows - 1
        self.cy = min(high, self.cy + max(1, count))
        self.wrap_next = False

    def cursor_forward(self, count: int = 1) -> None:
        self.cx = min(self.cols - 1, self.cx + max(1, count))
        self.wrap_next = False

    def cursor_back(self, count: int = 1) -> None:
        self.cx = max(0, self.cx - max(1, count))
        self.wrap_next = False

    def cursor_column(self, col: int = 1) -> None:
        self.cx = min(max(1, col), self.cols) - 1
        self.wrap_next = False

    def cursor_row(self, row: int = 1) -> None:
        self.cursor_position(row, self.cx + 1)

    def save_cursor(self) -> None:
        self.saved[self.in_alt_screen] = (self.cx, self.cy, self.attr, self.origin_mode,
                                          self.wrap_next, list(self.charsets), self.shift)

    def restore_cursor(self) -> None:
        saved = self.saved.get(self.in_alt_screen)
        if saved is None:
            self.cursor_position(1, 1)
            self.attr = PLAIN
            return
        cx, cy, self.attr, self.origin_mode, self.wrap_next, charsets, self.shift = saved
        self.charsets = list(charsets)
        self.cx, self.cy = min(cx, self.cols - 1), min(cy, self.rows - 1)
