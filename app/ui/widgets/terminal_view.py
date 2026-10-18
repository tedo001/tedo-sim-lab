"""A terminal widget: paints a :class:`~app.services.pty_session.PtySession`'s screen and sends
it the keyboard and mouse.

- Every key goes to the program, including the window's own shortcuts (Ctrl+B, Ctrl+K, …);
  Ctrl+` (the terminal panel) stays with the window.
- Select with the mouse (double-click: a word, triple-click: a line); copy with Ctrl+Shift+C
  (or Ctrl+C while something is selected), paste with Ctrl+Shift+V or Shift+Insert.
- The wheel scrolls the scrollback (Shift+PgUp/PgDn too); in full-screen programs it sends the
  arrow keys, or mouse events when the program asked for them (Shift keeps local selection).
- Ctrl+click opens a link in the browser, or a file (``path:line``) in the Code page.
- Ctrl+wheel or the context menu changes the font size.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QKeyEvent, QKeySequence, QPainter
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QMenu, QWidget

from ...services.pty_session import PtySession
from ..theme import SIZES
from .terminal_keys import encode
from .terminal_mouse import MouseMixin
from .terminal_paint import CellFonts, TerminalPalette, paint_cursor, paint_rows

__all__ = ["TerminalView"]

#: A link, or a file with an optional line: ``src/app.py:12``, ``File "x.py", line 3``.
_LINK = re.compile(r"https?://[^\s'\"<>]+")
_FILE = re.compile(r'File "(?P<path>[^"]+)", line (?P<tb>\d+)|(?P<p>[\w./\\~:-]+\.\w+)(?::(?P<line>\d+))?')
WORD = re.compile(r"[\w./\\~:@%+=#-]")


class TerminalView(MouseMixin, QAbstractScrollArea):
    open_path = Signal(Path, int)
    #: Enter was pressed after the program ended: start it again.
    restart_requested = Signal()

    def __init__(self, session: PtySession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.palette_ = TerminalPalette()
        self.fonts = CellFonts(SIZES["font_mono"] + 1)
        self.selection: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._anchor: tuple[int, int] | None = None
        self._top_line: int | None = None   # first line shown while scrolled back; None = live
        self._resize_timer = QTimer(self, singleShot=True, interval=40)
        self._resize_timer.timeout.connect(self.fit_to_size)
        self.setFrameShape(QAbstractScrollArea.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
        self.viewport().setAutoFillBackground(False)
        self.setObjectName("Terminal")
        self.setMinimumSize(160, 80)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)
        session.changed.connect(self._changed)
        self.init_mouse()
        self._sync_scrollbar()

    # Size --------------------------------------------------------------------------------------
    def grid_size(self) -> tuple[int, int]:
        """(rows, cols) that fit the viewport."""
        area = self.viewport().size()
        return (max(2, int(area.height() // self.fonts.height)),
                max(2, int(area.width() // self.fonts.width)))

    def fit_to_size(self) -> None:
        rows, cols = self.grid_size()
        self.session.resize(rows, cols)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._resize_timer.start()

    def zoom(self, steps: int) -> None:
        size = 0 if steps == 0 else min(28, max(9, self.fonts.pixel_size + steps))
        self.fonts = CellFonts(size or SIZES["font_mono"] + 1)
        self.fit_to_size()
        self.viewport().update()

    # Scrollback ------------------------------------------------------------------------------------
    def _sync_scrollbar(self) -> None:
        screen, bar = self.session.screen, self.verticalScrollBar()
        lines = 0 if screen.in_alt_screen else len(screen.scrollback)
        bar.blockSignals(True)
        bar.setRange(0, lines)
        bar.setPageStep(screen.rows)
        if self._top_line is None:
            bar.setValue(lines)
        else:
            first = screen.history_total - len(screen.scrollback)
            bar.setValue(min(lines, max(0, self._top_line - first)))
            if bar.value() >= lines:
                self._top_line = None
        bar.blockSignals(False)

    def _scrolled(self, value: int) -> None:
        screen = self.session.screen
        if value >= self.verticalScrollBar().maximum():
            self._top_line = None
        else:
            self._top_line = screen.history_total - len(screen.scrollback) + value
        self.viewport().update()

    def scroll_lines(self, lines: int) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() + lines)

    def scroll_to_bottom(self) -> None:
        self._top_line = None
        self._sync_scrollbar()
        self.viewport().update()

    def first_visible_line(self) -> int:
        screen = self.session.screen
        return screen.history_total if self._top_line is None else self._top_line

    def _changed(self) -> None:
        self._sync_scrollbar()
        self.viewport().update()

    # Painting ---------------------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        screen = self.session.screen
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), self.palette_.bg)
        first = self.first_visible_line()
        rows = [screen.history_line(first + y) for y in range(screen.rows)]
        paint_rows(painter, rows, self.fonts, self.palette_, first_line=first,
                   selection=self.selection, reverse_screen=screen.reverse_video)
        if (self._top_line is None and screen.cursor_visible and self.session.exit_code is None
                and 0 <= screen.cy < screen.rows):
            cell = screen.lines[screen.cy][min(screen.cx, screen.cols - 1)]
            paint_cursor(painter, self.fonts, self.palette_, x=screen.cx, y=screen.cy, cell=cell,
                         shape=screen.cursor_shape, focused=self.hasFocus())
        painter.end()

    # Keyboard -----------------------------------------------------------------------------------------------
    def event(self, event: QEvent) -> bool:
        # Keys that are window shortcuts elsewhere belong to the program while it has focus.
        if event.type() == QEvent.Type.ShortcutOverride and isinstance(event, QKeyEvent):
            if not _window_key(event) and (encode(event, self.session.screen) or _local_key(event)):
                event.accept()
                return True
        return super().event(event)

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: N802, A002 (Qt API)
        return False  # Tab goes to the program

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt API)
        mods = event.modifiers()
        ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        key = event.key()
        if (mods & ctrl_shift) == ctrl_shift and key in (Qt.Key.Key_C, Qt.Key.Key_V):
            self.copy() if key == Qt.Key.Key_C else self.paste()
            return
        if mods == Qt.KeyboardModifier.ShiftModifier and key == Qt.Key.Key_Insert:
            self.paste()
            return
        if mods == Qt.KeyboardModifier.ShiftModifier and key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self.scroll_lines((-1 if key == Qt.Key.Key_PageUp else 1) * (self.session.screen.rows - 1))
            return
        if event.matches(QKeySequence.StandardKey.Copy) and self.selection:
            self.copy()  # Ctrl+C copies while something is selected, otherwise interrupts
            self.selection = None
            self.viewport().update()
            return
        if self.session.exit_code is not None:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.restart_requested.emit()
            return
        data = encode(event, self.session.screen)
        if data:
            self.selection = None
            self.scroll_to_bottom()
            self.session.write(data)
        else:
            super().keyPressEvent(event)

    def inputMethodEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.commitString():
            self.session.write(event.commitString())
        event.accept()

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().focusInEvent(event)
        if self.session.screen.focus_events:
            self.session.write("\x1b[I")
        self.viewport().update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().focusOutEvent(event)
        if self.session.screen.focus_events:
            self.session.write("\x1b[O")
        self.viewport().update()

    # Clipboard ---------------------------------------------------------------------------------------
    def selected_text(self) -> str:
        if self.selection is None:
            return ""
        (start_line, start_col), (end_line, end_col) = self.selection
        screen, parts = self.session.screen, []
        for line in range(start_line, end_line + 1):
            row = screen.history_line(line)
            if row is None:
                continue
            first = start_col if line == start_line else 0
            last = end_col if line == end_line else len(row)
            text = "".join(cell[0] for cell in row[first:last])
            wrapped = getattr(row, "wrapped", False) and line != end_line
            parts.append(text if wrapped else text.rstrip() + ("\n" if line != end_line else ""))
        return "".join(parts)

    def copy(self) -> None:
        text = self.selected_text()
        if text:
            QGuiApplication.clipboard().setText(text)

    def paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        if text:
            self.scroll_to_bottom()
            self.session.paste(text)

    def select_all(self) -> None:
        screen = self.session.screen
        first = screen.history_total - len(screen.scrollback)
        self.selection = ((first, 0), (screen.history_total + screen.rows - 1, screen.cols))
        self.viewport().update()

    def clear_scrollback(self) -> None:
        self.session.screen.scrollback.clear()
        self.selection = None
        self.scroll_to_bottom()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.session.screen.mouse_mode and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return
        menu = QMenu(self)
        menu.addAction("Copy", self.copy).setEnabled(bool(self.selection))
        menu.addAction("Paste", self.paste)
        menu.addAction("Select all", self.select_all)
        menu.addSeparator()
        menu.addAction("Clear scrollback", self.clear_scrollback)
        menu.addSeparator()
        menu.addAction("Larger text", lambda: self.zoom(1))
        menu.addAction("Smaller text", lambda: self.zoom(-1))
        menu.addAction("Default text size", lambda: self.zoom(0))
        menu.exec(event.globalPos())

    # Links --------------------------------------------------------------------------------------------------
    def open_link_at(self, point: QPoint) -> bool:
        """Ctrl+click: a URL opens in the browser, an existing file asks the Code page to open it."""
        line, col = self.cell_at(point, exact=True)
        text, col = self._logical_line(line, col)
        if not text:
            return False
        for match in _LINK.finditer(text):
            if match.start() <= col < match.end():
                QDesktopServices.openUrl(QUrl(match.group(0).rstrip(".,;)")))
                return True
        for match in _FILE.finditer(text):
            if match.start() <= col < match.end():
                name = match.group("path") or match.group("p")
                number = int(match.group("tb") or match.group("line") or 1)
                path = Path(name).expanduser()
                path = path if path.is_absolute() else self.session.cwd / path
                if path.is_file():
                    self.open_path.emit(path, number)
                    return True
        return False


    def _logical_line(self, line: int, col: int) -> tuple[str, int]:
        """The text of the line ``line`` belongs to, wrapped rows joined, and where ``col`` is in it."""
        screen = self.session.screen
        if screen.history_line(line) is None:
            return "", 0
        first = line
        def wrapped(number: int) -> bool:
            return getattr(screen.history_line(number), "wrapped", False)

        while wrapped(first - 1):
            first -= 1
        last = line
        while wrapped(last) and screen.history_line(last + 1) is not None:
            last += 1
        rows = [screen.history_line(n) for n in range(first, last + 1)]
        offset = sum(len(row) for row in rows[:line - first]) + col
        return "".join(cell[0] or " " for row in rows for cell in row), offset


def _window_key(event: QKeyEvent) -> bool:
    """Keys the window keeps even while a terminal has focus: Ctrl+` (the terminal panel)."""
    ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
    return event.key() == Qt.Key.Key_QuoteLeft and ctrl


def _local_key(event: QKeyEvent) -> bool:
    ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    return (event.modifiers() & ctrl_shift) == ctrl_shift and event.key() in (Qt.Key.Key_C, Qt.Key.Key_V)


def double_click_interval() -> int:
    return QApplication.doubleClickInterval()
