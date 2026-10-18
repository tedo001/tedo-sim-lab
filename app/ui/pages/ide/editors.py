"""Open files as editor tabs: load (text only, up to 5 MB), save (Python is syntax-checked on
save and the error line marked), unsaved marker, close with a question when there are unsaved
changes, follow renames and deletions from the tree, breakpoints kept per file for the session
(closing a tab keeps them), and a find / replace bar."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....services.project import check_python_syntax
from ...widgets import label
from ...widgets.code_editor import COMMENT, CodeEditor
from ...widgets.highlighters import highlighter_for

__all__ = ["EditorTabs", "FindBar", "MAX_BYTES"]

MAX_BYTES = 5_000_000


def read_text(path: Path) -> str:
    """The file's text; ``ValueError`` for a binary or very large file."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{path.name} is {size / 1e6:.1f} MB; the editor opens files up to "
                         f"{MAX_BYTES / 1e6:.0f} MB")
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise ValueError(f"{path.name} is not a text file")
    return data.decode("utf-8", errors="replace")


class EditorTabs(QTabWidget):
    current_file = Signal(object)  # Path or None
    message = Signal(str)
    breakpoint_toggled = Signal(Path, int, bool)  # file, line, set

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EditorTabs")
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.currentChanged.connect(lambda _: self.current_file.emit(self.current_path()))
        self._highlighters: dict[int, object] = {}
        self._breakpoints: dict[Path, set[int]] = {}

    # Tabs ----------------------------------------------------------------------------------------
    def editors(self) -> list[CodeEditor]:
        return [self.widget(i) for i in range(self.count())]

    def path_of(self, editor: CodeEditor) -> Path:
        return editor.property("path")

    def current_editor(self) -> CodeEditor | None:
        widget = self.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def current_path(self) -> Path | None:
        editor = self.current_editor()
        return self.path_of(editor) if editor else None

    def index_of(self, path: Path) -> int:
        for i, editor in enumerate(self.editors()):
            if self.path_of(editor) == path:
                return i
        return -1

    def open(self, path: Path) -> CodeEditor | None:
        path = path.resolve()
        index = self.index_of(path)
        if index >= 0:
            self.setCurrentIndex(index)
            return self.widget(index)
        try:
            text = read_text(path)
        except (ValueError, OSError) as exc:
            self.message.emit(str(exc))
            return None
        editor = CodeEditor(comment=COMMENT.get(path.suffix.lower(), "#"))
        editor.setProperty("path", path)
        editor.setPlainText(text)
        editor.document().setModified(False)
        editor.set_breakpoints(self._breakpoints.get(path, ()))
        editor.breakpoints_changed.connect(lambda e=editor: self._remember(e))
        editor.breakpoint_toggled.connect(
            lambda line, on, e=editor: self.breakpoint_toggled.emit(self.path_of(e), line, on))
        self._highlighters[id(editor)] = highlighter_for(path, editor.document())
        editor.document().modificationChanged.connect(lambda _m, e=editor: self._retitle(e))
        index = self.addTab(editor, path.name)
        self.setTabToolTip(index, str(path))
        self.setCurrentIndex(index)
        editor.setFocus()
        return editor

    def _retitle(self, editor: CodeEditor) -> None:
        index = self.indexOf(editor)
        if index >= 0:
            name = self.path_of(editor).name
            self.setTabText(index, f"● {name}" if editor.document().isModified() else name)

    def unsaved(self) -> list[Path]:
        return [self.path_of(e) for e in self.editors() if e.document().isModified()]

    # Breakpoints ---------------------------------------------------------------------------------
    def _remember(self, editor: CodeEditor) -> None:
        path = self.path_of(editor)
        if editor.breakpoints:
            self._breakpoints[path] = set(editor.breakpoints)
        else:
            self._breakpoints.pop(path, None)

    def breakpoints(self) -> dict[Path, set[int]]:
        """Every file's breakpoints, open in a tab or not."""
        return {path: set(lines) for path, lines in self._breakpoints.items() if lines}

    def show_debug_line(self, path: Path | None, line: int = 0) -> CodeEditor | None:
        """Mark where the debugger stopped (opening the file, keeping the keyboard where it is);
        ``None`` clears every mark."""
        for editor in self.editors():
            editor.set_debug_line(None)
        if path is None:
            return None
        focused = self.window().focusWidget() if self.window() else None
        editor = self.open(path)
        if editor is not None:
            editor.set_debug_line(line)
            editor.go_to_line(line, focus=False)
        if focused is not None:
            focused.setFocus()
        return editor

    # Saving --------------------------------------------------------------------------------------
    def save(self, editor: CodeEditor | None = None) -> bool:
        editor = editor or self.current_editor()
        if editor is None:
            return False
        path = self.path_of(editor)
        text = editor.toPlainText()
        try:
            path.write_text(text, encoding="utf-8", newline="")
        except OSError as exc:
            self.message.emit(f"Could not save {path.name}: {exc}")
            return False
        editor.document().setModified(False)
        problem = check_python_syntax(text, str(path)) if path.suffix.lower() in (".py", ".pyw") else None
        editor.set_error_line(problem[0] if problem else None)
        self.message.emit(f"Saved {path.name}" + (f" · syntax error on line {problem[0]}: {problem[1]}"
                                                   if problem else ""))
        return True

    def save_all(self) -> int:
        return sum(self.save(e) for e in self.editors() if e.document().isModified())

    def close_tab(self, index: int, *, ask: bool = True) -> bool:
        editor = self.widget(index)
        if editor is None:
            return False
        if ask and editor.document().isModified():
            answer = QMessageBox.question(
                self, "Unsaved changes", f"Save the changes to {self.path_of(editor).name}?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save and not self.save(editor):
                return False
        self.removeTab(index)
        self._highlighters.pop(id(editor), None)
        editor.deleteLater()
        return True

    def follow(self, old: Path, new: Path | None) -> None:
        """A file or folder was renamed (``new``) or moved to the trash (``None``)."""
        old = old.resolve()
        for editor in list(self.editors()):
            path = self.path_of(editor)
            if path != old and old not in path.parents:
                continue
            lines = self._breakpoints.pop(path, None)
            if new is None:
                self.close_tab(self.indexOf(editor), ask=False)
            else:
                moved = new.resolve() / path.relative_to(old) if path != old else new.resolve()
                if lines:
                    self._breakpoints[moved] = lines
                editor.setProperty("path", moved)
                self.setTabToolTip(self.indexOf(editor), str(moved))
                self._retitle(editor)
        self.current_file.emit(self.current_path())


class FindBar(QWidget):
    """Find and replace in the current editor (Ctrl+F shows it, Esc hides it)."""

    def __init__(self, tabs: EditorTabs, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tabs = tabs
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 4, 0, 4)
        row = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find")
        self.find.returnPressed.connect(self.next)
        self.replace = QLineEdit()
        self.replace.setPlaceholderText("Replace with")
        self.case = QCheckBox("Match case")
        row.addWidget(self.find, 2)
        for text, slot in (("Next", self.next), ("Previous", lambda: self.next(backwards=True))):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addWidget(self.replace, 2)
        replace_all = QPushButton("Replace all")
        replace_all.clicked.connect(self.replace_all)
        row.addWidget(replace_all)
        row.addWidget(self.case)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        row.addWidget(close)
        box.addLayout(row)
        self.result = label("", "CardCaption")
        box.addWidget(self.result)

    def show_for(self, text: str = "") -> None:
        self.show()
        if text:
            self.find.setText(text)
        self.find.setFocus()
        self.find.selectAll()

    def next(self, backwards: bool = False) -> bool:
        editor = self.tabs.current_editor()
        found = bool(editor and editor.find_text(self.find.text(), backwards=backwards,
                                                 case=self.case.isChecked()))
        self.result.setText("" if found or not self.find.text() else "Not found")
        return found

    def replace_all(self) -> int:
        editor = self.tabs.current_editor()
        count = editor.replace_all(self.find.text(), self.replace.text(), case=self.case.isChecked()) \
            if editor else 0
        self.result.setText(f"Replaced {count}" if count else "Not found")
        return count
