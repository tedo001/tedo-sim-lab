"""The project's files as a tree: open on double-click; new file, new folder, rename, move to
trash and show in the file manager from the right-click menu. Caches and environments
(``__pycache__``, ``.git``, ``.venv`` …) are hidden."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDir, QModelIndex, QSortFilterProxyModel, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileSystemModel, QInputDialog, QMenu, QMessageBox, QTreeView, QWidget

from ....services.project import safe_child, trash

__all__ = ["HIDDEN", "ProjectTree"]

HIDDEN = {"__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
          "node_modules", ".idea", ".ipynb_checkpoints"}


class _Hide(QSortFilterProxyModel):
    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802 (Qt API)
        name = self.sourceModel().index(row, 0, parent).data()
        return not (name in HIDDEN or str(name).startswith(".venv"))


class ProjectTree(QTreeView):
    open_requested = Signal(Path)
    #: (old path, new path or None when deleted) so open editors can follow
    moved = Signal(Path, object)
    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectTree")
        self.files = QFileSystemModel(self)
        self.files.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot | QDir.Filter.Hidden)
        self.files.setReadOnly(True)
        self.proxy = _Hide(self)
        self.proxy.setSourceModel(self.files)
        self.setModel(self.proxy)
        for column in (1, 2, 3):
            self.hideColumn(column)
        self.setHeaderHidden(True)
        self.setAnimated(False)
        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.doubleClicked.connect(self._activated)
        self.root: Path | None = None

    def set_root(self, folder: Path) -> None:
        self.root = folder
        source = self.files.setRootPath(str(folder))
        self.setRootIndex(self.proxy.mapFromSource(source))

    def path_at(self, index: QModelIndex) -> Path | None:
        if not index.isValid():
            return None
        return Path(self.files.filePath(self.proxy.mapToSource(index)))

    def select_path(self, path: Path) -> None:
        index = self.proxy.mapFromSource(self.files.index(str(path)))
        if index.isValid():
            self.setCurrentIndex(index)
            self.scrollTo(index)

    def _activated(self, index: QModelIndex) -> None:
        path = self.path_at(index)
        if path is not None and path.is_file():
            self.open_requested.emit(path)

    # Menu --------------------------------------------------------------------------------------
    def _target_folder(self, path: Path | None) -> Path:
        if path is None:
            return self.root or Path.home()
        return path if path.is_dir() else path.parent

    def _menu(self, point) -> None:  # noqa: ANN001 (QPoint)
        path = self.path_at(self.indexAt(point))
        menu = QMenu(self)
        menu.addAction("New file…", lambda: self.new_file(self._target_folder(path)))
        menu.addAction("New folder…", lambda: self.new_folder(self._target_folder(path)))
        if path is not None:
            menu.addSeparator()
            if path.is_file():
                menu.addAction("Open", lambda: self.open_requested.emit(path))
            menu.addAction("Rename…", lambda: self.rename(path))
            menu.addAction("Move to trash…", lambda: self.delete(path))
        menu.addSeparator()
        folder = self._target_folder(path)
        menu.addAction("Show in file manager",
                       lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))
        menu.exec(self.viewport().mapToGlobal(point))

    def _ask(self, title: str, label: str, text: str = "") -> str | None:
        value, ok = QInputDialog.getText(self, title, label, text=text)
        return value.strip() if ok and value.strip() else None

    def new_file(self, folder: Path, name: str | None = None) -> Path | None:
        name = name or self._ask("New file", f"File name in {folder.name}:", "untitled.py")
        if not name:
            return None
        try:
            path = safe_child(folder, name)
            if path.exists():
                raise ValueError(f"{name} already exists")
            path.touch()
        except (ValueError, OSError) as exc:
            self.message.emit(f"Could not create the file: {exc}")
            return None
        self.select_path(path)
        self.open_requested.emit(path)
        return path

    def new_folder(self, folder: Path, name: str | None = None) -> Path | None:
        name = name or self._ask("New folder", f"Folder name in {folder.name}:")
        if not name:
            return None
        try:
            path = safe_child(folder, name)
            path.mkdir()
        except (ValueError, OSError) as exc:
            self.message.emit(f"Could not create the folder: {exc}")
            return None
        self.select_path(path)
        return path

    def rename(self, path: Path, name: str | None = None) -> Path | None:
        name = name or self._ask("Rename", "New name:", path.name)
        if not name or name == path.name:
            return None
        try:
            target = safe_child(path.parent, name)
            if target.exists():
                raise ValueError(f"{name} already exists")
            path.rename(target)
        except (ValueError, OSError) as exc:
            self.message.emit(f"Could not rename: {exc}")
            return None
        self.moved.emit(path, target)
        return target

    def delete(self, path: Path, confirm: bool = True) -> bool:
        if confirm and QMessageBox.question(self, "Move to trash", f"Move {path.name} to the trash?") \
                != QMessageBox.StandardButton.Yes:
            return False
        problem = trash(path)
        if problem:
            self.message.emit(problem)
            return False
        self.moved.emit(path, None)
        return True
