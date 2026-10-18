"""The terminal widget's mouse: local selection (drag, double-click word, triple-click line),
Ctrl+click links, the wheel (scrollback, arrow keys in full-screen programs, zoom with Ctrl),
and xterm mouse reporting (X10, normal, button and any-motion tracking; SGR or legacy coordinates)
for programs that ask for it — Shift always keeps the mouse local."""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

__all__ = ["MouseMixin"]

_BUTTONS = {Qt.MouseButton.LeftButton: 0, Qt.MouseButton.MiddleButton: 1, Qt.MouseButton.RightButton: 2}


class MouseMixin:
    """Mixed into :class:`~app.ui.widgets.terminal_view.TerminalView`."""

    def init_mouse(self) -> None:
        self._last_double = 0.0
        self._pressed_button: int | None = None
        self._last_report_cell: tuple[int, int] | None = None
        self.viewport().setMouseTracking(True)

    def cell_at(self, point: QPoint, *, exact: bool = False) -> tuple[int, int]:
        """(line number, column) under ``point``; with ``exact`` the cell, otherwise the nearest
        boundary between cells (what a selection edge snaps to)."""
        screen = self.session.screen
        y = min(screen.rows - 1, max(0, int(point.y() // self.fonts.height)))
        position = point.x() / self.fonts.width
        col = int(position) if exact else int(position + 0.5)
        return self.first_visible_line() + y, min(screen.cols - (1 if exact else 0), max(0, col))

    def _reporting(self, event) -> bool:
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        return bool(self.session.screen.mouse_mode) and not shift and self._top_line is None

    def _report(self, code: int, point: QPoint, press: bool, modifiers) -> None:
        screen = self.session.screen
        col = min(screen.cols, max(1, int(point.x() // self.fonts.width) + 1))
        row = min(screen.rows, max(1, int(point.y() // self.fonts.height) + 1))
        if screen.mouse_mode != 9:
            if modifiers & Qt.KeyboardModifier.AltModifier:
                code += 8
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                code += 16
        if screen.mouse_sgr:
            self.session.write(f"\x1b[<{code};{col};{row}{'M' if press else 'm'}")
        else:
            if not press:
                code = 3 | (code & ~3 & ~64)
            if col <= 223 and row <= 223:
                self.session.write(f"\x1b[M{chr(32 + code)}{chr(32 + col)}{chr(32 + row)}")

    # Buttons -----------------------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.setFocus()
        point = event.position().toPoint()
        if self._reporting(event) and event.button() in _BUTTONS:
            self._pressed_button = _BUTTONS[event.button()]
            self._report(self._pressed_button, point, True, event.modifiers())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and self.open_link_at(point):
            return
        if time.monotonic() - self._last_double < QApplication.doubleClickInterval() / 1000:
            line, _col = self.cell_at(point, exact=True)  # third click: the whole line
            self.selection = ((line, 0), (line, self.session.screen.cols))
            self._anchor = None
        else:
            self._anchor = self.cell_at(point)
            self.selection = None
        self.viewport().update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        point = event.position().toPoint()
        screen = self.session.screen
        if self._reporting(event):
            held = self._pressed_button is not None
            moving = screen.mouse_mode == 1003 or (screen.mouse_mode == 1002 and held)
            cell = (int(point.x() // self.fonts.width), int(point.y() // self.fonts.height))
            if moving and cell != self._last_report_cell:
                self._last_report_cell = cell
                button = self._pressed_button if self._pressed_button is not None else 3
                self._report(32 + button, point, True, event.modifiers())
            return
        if self._anchor is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        if point.y() < 0:
            self.scroll_lines(-1)
        elif point.y() > self.viewport().height():
            self.scroll_lines(1)
        here = self.cell_at(point)
        start, end = sorted((self._anchor, here))
        self.selection = (start, end) if start != end else None
        self.viewport().update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        point = event.position().toPoint()
        if self._pressed_button is not None and self.session.screen.mouse_mode:
            if self.session.screen.mouse_mode != 9:
                self._report(self._pressed_button, point, False, event.modifiers())
            self._pressed_button = None
            return
        self._pressed_button = None
        self._anchor = None
        if event.button() == Qt.MouseButton.MiddleButton:
            clipboard = QApplication.clipboard()
            text = clipboard.text(clipboard.Mode.Selection) if clipboard.supportsSelection() else ""
            if text:
                self.session.paste(text)
        elif self.selection and QApplication.clipboard().supportsSelection():
            QApplication.clipboard().setText(self.selected_text(), QApplication.clipboard().Mode.Selection)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        if self._reporting(event) or event.button() != Qt.MouseButton.LeftButton:
            return
        from .terminal_view import WORD

        line, col = self.cell_at(event.position().toPoint(), exact=True)
        row = self.session.screen.history_line(line)
        if row is None:
            return
        chars = [cell[0] for cell in row]
        if not chars or not WORD.match(chars[col] or " "):
            return
        start = col
        while start > 0 and WORD.match(chars[start - 1] or " "):
            start -= 1
        end = col
        while end < len(chars) and WORD.match(chars[end] or " "):
            end += 1
        self.selection = ((line, start), (line, end))
        self._anchor = None
        self._last_double = time.monotonic()
        self.viewport().update()

    # The wheel -------------------------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt API)
        steps = event.angleDelta().y() / 120
        if not steps:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1 if steps > 0 else -1)
            return
        screen = self.session.screen
        if self._reporting(event):
            for _ in range(int(abs(steps)) or 1):
                self._report(64 if steps > 0 else 65, event.position().toPoint(), True, event.modifiers())
            return
        if screen.in_alt_screen and screen.alt_scroll:
            key = ("\x1bO" if screen.app_cursor else "\x1b[") + ("A" if steps > 0 else "B")
            self.session.write(key * (3 * (int(abs(steps)) or 1)))
            return
        self.scroll_lines(int(-3 * steps) or (-1 if steps > 0 else 1))
