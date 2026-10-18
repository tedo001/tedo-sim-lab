"""A plain-text code editor: line numbers, the current line marked, an error line marked,
breakpoints (click the gutter, or F9) that move with the text, the line a debugger stopped at,
indentation that follows the previous line, Tab as four spaces, Shift+Tab to dedent,
Ctrl+/ to comment or uncomment, and find/replace helpers for the IDE's find bar."""

from __future__ import annotations

import re

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget

from ..theme import mono_font
from ..theme.tokens import COLORS

__all__ = ["CodeEditor"]

INDENT = "    "
#: Comment marker by file suffix (Ctrl+/).
COMMENT = {".py": "#", ".pyw": "#", ".yaml": "#", ".yml": "#", ".toml": "#", ".sh": "#", ".r": "#",
           ".ini": ";", ".cfg": "#", ".ps1": "#", ".bat": "REM", ".js": "//", ".ts": "//",
           ".c": "//", ".cpp": "//", ".h": "//", ".java": "//", ".sql": "--", ".md": "<!--"}


def _dedent(text: str) -> str:
    """One indentation level less: four spaces, a tab, or whatever leading spaces there are."""
    if text.startswith(INDENT):
        return text[len(INDENT):]
    if text.startswith("\t"):
        return text[1:]
    return text.lstrip(" ")


class _Gutter(QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        self.editor.paint_gutter(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            block = self.editor.cursorForPosition(QPoint(0, round(event.position().y()))).block()
            self.editor.toggle_breakpoint(block.blockNumber() + 1)


class CodeEditor(QPlainTextEdit):
    #: The set of breakpoint lines changed (toggled, or moved by an edit).
    breakpoints_changed = Signal()
    #: The person set (True) or cleared (False) the breakpoint on a line.
    breakpoint_toggled = Signal(int, bool)

    def __init__(self, parent: QWidget | None = None, *, comment: str = "#") -> None:
        super().__init__(parent)
        self.setObjectName("CodeEditor")
        self.setFont(mono_font(13))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.comment = comment
        self.error_line: int | None = None
        self.debug_line: int | None = None
        self.breakpoints: set[int] = set()
        self._lines = self.blockCount()
        self._gutter = _Gutter(self)
        self._gutter.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gutter.setToolTip("Click to set or clear a breakpoint (F9)")
        self.document().contentsChange.connect(self._move_breakpoints)
        self.blockCountChanged.connect(lambda _: self._fit_gutter())
        self.updateRequest.connect(self._scroll_gutter)
        self.cursorPositionChanged.connect(self._mark_lines)
        self._fit_gutter()
        self._mark_lines()

    # Gutter ---------------------------------------------------------------------------------
    def gutter_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 18 + self._dot_space() + self.fontMetrics().horizontalAdvance("9") * max(3, digits)

    def _dot_space(self) -> int:
        return self.fontMetrics().height()

    def _fit_gutter(self) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _scroll_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        area = self.contentsRect()
        self._gutter.setGeometry(QRect(area.left(), area.top(), self.gutter_width(), area.height()))

    def paint_gutter(self, event: QPaintEvent) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(COLORS["surface"]))
        painter.setFont(self.font())
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        current = self.textCursor().blockNumber()
        height, dot = self.fontMetrics().height(), self._dot_space()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        while block.isValid() and top <= event.rect().bottom():
            bottom = top + round(self.blockBoundingRect(block).height())
            if block.isVisible() and bottom >= event.rect().top():
                if number + 1 in self.breakpoints:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(COLORS["code_breakpoint"]))
                    size = max(6, dot - 8)
                    painter.drawEllipse(4 + (dot - size) // 2, top + (height - size) // 2, size, size)
                colour = (COLORS["code_error"] if number + 1 == self.error_line else
                          COLORS["text"] if number in (current, (self.debug_line or 0) - 1) else
                          COLORS["code_gutter"])
                painter.setPen(QColor(colour))
                painter.drawText(0, top, self._gutter.width() - 8, height,
                                 Qt.AlignmentFlag.AlignRight, str(number + 1))
            block, top, number = block.next(), bottom, number + 1
        painter.end()

    # Breakpoints ------------------------------------------------------------------------------
    def toggle_breakpoint(self, line: int | None = None) -> None:
        """Set or clear a breakpoint on ``line`` (the cursor's line by default)."""
        line = line or self.textCursor().blockNumber() + 1
        if not 1 <= line <= self.blockCount():
            return
        self.breakpoints ^= {line}
        self._gutter.update()
        self.breakpoints_changed.emit()
        self.breakpoint_toggled.emit(line, line in self.breakpoints)

    def set_breakpoints(self, lines) -> None:  # noqa: ANN001 (iterable of int)
        self.breakpoints = {line for line in lines if 1 <= line <= self.blockCount()}
        self._gutter.update()
        self.breakpoints_changed.emit()

    def _move_breakpoints(self, position: int, _removed: int, _added: int) -> None:
        """Lines added or removed above a breakpoint move it; a removed line takes its own."""
        delta, self._lines = self.blockCount() - self._lines, self.blockCount()
        if not delta or not self.breakpoints:
            return
        block = self.document().findBlock(position)
        # The first line whose text moved: the edit's own line when it began at the line's start.
        first = block.blockNumber() + (1 if position == block.position() else 2)
        gone = range(first, first - delta)  # lines removed (none when lines were added)
        moved = {line + delta if line >= first else line for line in self.breakpoints if line not in gone}
        moved = {line for line in moved if 1 <= line <= self.blockCount()}
        if moved != self.breakpoints:
            self.breakpoints = moved
            self.breakpoints_changed.emit()

    def set_debug_line(self, line: int | None) -> None:
        """Mark the line a debugger stopped at (``None`` clears it)."""
        self.debug_line = line
        self._mark_lines()

    # Marks ----------------------------------------------------------------------------------
    def _mark_lines(self) -> None:
        marks = []
        current = QTextEdit.ExtraSelection()
        current.format.setBackground(QColor(COLORS["code_line"]))
        current.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        current.cursor = self.textCursor()
        current.cursor.clearSelection()
        marks.append(current)
        if self.debug_line is not None:
            block = self.document().findBlockByNumber(self.debug_line - 1)
            if block.isValid():
                stopped = QTextEdit.ExtraSelection()
                stopped.format.setBackground(QColor(COLORS["code_debug_line"]))
                stopped.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                stopped.cursor = QTextCursor(block)
                marks.append(stopped)
        if self.error_line is not None:
            block = self.document().findBlockByNumber(self.error_line - 1)
            if block.isValid():
                error = QTextEdit.ExtraSelection()
                error.format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                error.format.setUnderlineColor(QColor(COLORS["code_error"]))
                error.cursor = QTextCursor(block)
                error.cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                marks.append(error)
        self.setExtraSelections(marks)
        self._gutter.update()

    def set_error_line(self, line: int | None) -> None:
        self.error_line = line
        self._mark_lines()

    def go_to_line(self, line: int, column: int = 0, *, focus: bool = True) -> None:
        block = self.document().findBlockByNumber(max(0, line - 1))
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.Right, n=max(0, min(column, block.length() - 1)))
        self.setTextCursor(cursor)
        self.centerCursor()
        if focus:
            self.setFocus()

    # Editing ----------------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key, mods = event.key(), event.modifiers()
        if key == Qt.Key.Key_Tab and not mods:
            if self.textCursor().hasSelection():
                self.indent_selection(+1)
            else:
                self.insertPlainText(INDENT)
            return
        if key == Qt.Key.Key_Backtab:
            self.indent_selection(-1)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not mods:
            line = self.textCursor().block().text()[:self.textCursor().positionInBlock()]
            indent = line[:len(line) - len(line.lstrip())]
            if line.rstrip().endswith(":") and self.comment == "#":
                indent += INDENT
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        if key == Qt.Key.Key_Slash and mods == Qt.KeyboardModifier.ControlModifier:
            self.toggle_comment()
            return
        if key == Qt.Key.Key_F9 and not mods:
            self.toggle_breakpoint()
            return
        super().keyPressEvent(event)

    def _selected_blocks(self) -> tuple[int, int]:
        cursor = self.textCursor()
        document = self.document()
        first = document.findBlock(cursor.selectionStart()).blockNumber()
        last_position = cursor.selectionEnd()
        last = document.findBlock(last_position).blockNumber()
        at_line_start = document.findBlock(last_position).position() == last_position
        if cursor.hasSelection() and at_line_start and last > first:
            last -= 1  # a selection ending at a line start does not include that line
        return first, last

    def _edit_lines(self, change) -> None:  # noqa: ANN001 (str -> str)
        first, last = self._selected_blocks()
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        for number in range(first, last + 1):
            block = self.document().findBlockByNumber(number)
            new = change(block.text())
            if new != block.text():
                cursor.setPosition(block.position())
                cursor.setPosition(block.position() + len(block.text()),
                                   QTextCursor.MoveMode.KeepAnchor)
                cursor.insertText(new)
        cursor.endEditBlock()

    def indent_selection(self, direction: int) -> None:
        if direction > 0:
            self._edit_lines(lambda text: INDENT + text if text.strip() else text)
        else:
            self._edit_lines(_dedent)

    def toggle_comment(self) -> None:
        marker = self.comment
        first, last = self._selected_blocks()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        filled = [line for line in lines if line.strip()]
        uncomment = bool(filled) and all(line.lstrip().startswith(marker) for line in filled)

        def change(text: str) -> str:
            if not text.strip():
                return text
            indent = text[:len(text) - len(text.lstrip())]
            body = text.lstrip()
            if uncomment:
                body = body[len(marker):]
                return indent + (body[1:] if body.startswith(" ") else body)
            return f"{indent}{marker} {body}"

        self._edit_lines(change)

    # Find -------------------------------------------------------------------------------------
    def find_text(self, text: str, *, backwards: bool = False, case: bool = False) -> bool:
        """Select the next match (wrapping around); ``False`` when there is none."""
        if not text:
            return False
        flags = QTextDocument.FindFlag(0)
        if backwards:
            flags |= QTextDocument.FindFlag.FindBackward
        if case:
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if self.find(text, flags):
            return True
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End if backwards
                            else QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        return self.find(text, flags)

    def replace_all(self, text: str, replacement: str, *, case: bool = False) -> int:
        if not text:
            return 0
        source = self.toPlainText()
        count = source.count(text) if case else source.lower().count(text.lower())
        if not count:
            return 0
        new = re.sub(re.escape(text), lambda _m: replacement, source, flags=0 if case else re.IGNORECASE)
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(new)
        cursor.endEditBlock()
        return count
