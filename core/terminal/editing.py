"""What a :class:`~core.terminal.screen.Screen` does besides printing and moving the cursor:
erasing, inserting and deleting characters and lines, the alternate screen, resizing, and
reading lines back (for the widget, copying and tests)."""

from __future__ import annotations

from .cells import PLAIN, Row

__all__ = ["EditingMixin"]


class EditingMixin:
    """Mixed into :class:`~core.terminal.screen.Screen`; uses its cursor, lines and modes."""

    # Erasing and editing -------------------------------------------------------------------------
    def erase_line(self, mode: int = 0) -> None:
        row, blank = self.lines[self.cy], self._blank()
        start, end = {0: (self.cx, self.cols), 1: (0, self.cx + 1)}.get(mode, (0, self.cols))
        row[start:end] = [blank] * (end - start)
        if mode != 1:
            row.wrapped = False
        self.wrap_next = False

    def erase_display(self, mode: int = 0) -> None:
        blank = self._blank()
        if mode == 3:  # xterm: forget the scrollback
            self.scrollback.clear()
            return
        if mode == 0:
            self.erase_line(0)
            rows = range(self.cy + 1, self.rows)
        elif mode == 1:
            self.erase_line(1)
            rows = range(0, self.cy)
        else:
            rows = range(self.rows)
        for y in rows:
            self.lines[y] = self._blank_row(blank)

    def erase_chars(self, count: int = 1) -> None:
        count = min(max(1, count), self.cols - self.cx)
        self.lines[self.cy][self.cx:self.cx + count] = [self._blank()] * count
        self.wrap_next = False

    def delete_chars(self, count: int = 1) -> None:
        row = self.lines[self.cy]
        count = min(max(1, count), self.cols - self.cx)
        del row[self.cx:self.cx + count]
        row.extend([self._blank()] * count)
        self.wrap_next = False

    def insert_chars(self, count: int = 1) -> None:
        row = self.lines[self.cy]
        count = min(max(1, count), self.cols - self.cx)
        row[self.cx:self.cx] = [self._blank()] * count
        del row[self.cols:]
        self.wrap_next = False

    def insert_lines(self, count: int = 1) -> None:
        if not self.top <= self.cy <= self.bottom:
            return
        for _ in range(min(max(1, count), self.bottom - self.cy + 1)):
            del self.lines[self.bottom]
            self.lines.insert(self.cy, self._blank_row(self._blank()))
        self.carriage_return()

    def delete_lines(self, count: int = 1) -> None:
        if not self.top <= self.cy <= self.bottom:
            return
        for _ in range(min(max(1, count), self.bottom - self.cy + 1)):
            del self.lines[self.cy]
            self.lines.insert(self.bottom, self._blank_row(self._blank()))
        self.carriage_return()

    def alignment_test(self) -> None:
        """DECALN: fill the screen with E (used by terminal test programs)."""
        for y in range(self.rows):
            self.lines[y] = Row([("E", PLAIN)] * self.cols)
        self.cursor_position(1, 1)

    # The alternate screen --------------------------------------------------------------------------
    def enter_alt_screen(self, *, clear: bool = True) -> None:
        if self.alt is None or clear:
            self.alt = [self._blank_row() for _ in range(self.rows)]
        self.lines = self.alt
        self.top, self.bottom = 0, self.rows - 1

    def leave_alt_screen(self) -> None:
        self.lines = self.main
        self.top, self.bottom = 0, self.rows - 1

    # Size -----------------------------------------------------------------------------------------------
    def resize(self, rows: int, cols: int) -> None:
        """New size. When the screen shrinks, blank lines below the cursor go first, then lines
        above it scroll into the scrollback; when it grows, lines come back from the scrollback."""
        rows, cols = max(2, rows), max(2, cols)
        if (rows, cols) == (self.rows, self.cols):
            return
        for buffer in (self.main, self.alt):
            if buffer is None:
                continue
            for y, row in enumerate(buffer):
                buffer[y] = self._fit(row, cols)
            self._resize_rows(buffer, rows, cols, main=buffer is self.main)
        self.rows, self.cols = rows, cols
        self.top, self.bottom = 0, rows - 1
        self.tabs = set(range(8, cols, 8))
        self.cx, self.cy = min(self.cx, cols - 1), min(max(0, self.cy), rows - 1)
        self.wrap_next = False

    @staticmethod
    def _fit(row: list, cols: int) -> Row:
        fitted = Row(row[:cols] + [(" ", PLAIN)] * (cols - len(row)))
        fitted.wrapped = getattr(row, "wrapped", False)
        return fitted

    def _resize_rows(self, buffer: list[Row], rows: int, cols: int, *, main: bool) -> None:
        live = buffer is self.lines
        while len(buffer) > rows:
            if live and self.cy >= rows:  # the cursor would fall off: the top scrolls away
                gone = buffer.pop(0)
                if main and self.scrollback.maxlen:
                    self.scrollback.append(gone)
                    self.history_total += 1
                self.cy -= 1
            else:
                buffer.pop()
        while len(buffer) < rows:
            if main and live and self.scrollback:
                buffer.insert(0, self._fit(self.scrollback.pop(), cols))
                self.history_total -= 1
                self.cy += 1
            else:
                buffer.append(Row([(" ", PLAIN)] * cols))

    # Reading the screen (the widget, copy, tests) ----------------------------------------------------
    @staticmethod
    def row_text(row: list) -> str:
        return "".join(cell[0] for cell in row)

    def display(self) -> list[str]:
        """The visible lines as text (wide characters once, trailing blanks kept)."""
        return [self.row_text(row) for row in self.lines]

    def history_line(self, number: int) -> Row | None:
        """Line ``number`` in the stable numbering: scrollback lines, then the screen's."""
        first = self.history_total - len(self.scrollback)
        if number < first:
            return None
        if number < self.history_total:
            return self.scrollback[number - first]
        y = number - self.history_total
        return self.lines[y] if 0 <= y < self.rows else None
