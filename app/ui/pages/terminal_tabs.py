"""Terminal tabs: the Terminal page, the bottom panel (Ctrl+`) and the Code page all use this.

A "+" menu lists every profile (shells, the experiment Python, vim, Neovim, Docker, a command of
your own); ones that are not installed are shown disabled with the reason. When a program ends,
its tab says so and Enter starts it again. Closing a tab whose program is still busy asks first.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.terminal import Profile, PtyError, default_profile, find_profile

from ...services.context import AppContext
from ...services.pty_session import PtySession
from ..icons import icon
from ..widgets import PathLabel, label
from ..widgets.terminal_view import TerminalView
from .terminal_dialogs import DockerDialog, ask_command

__all__ = ["TerminalTabs"]


class TerminalTabs(QWidget):
    open_path = Signal(Path, int)
    #: A finished tab started again: (old view, new view).
    restarted = Signal(object, object)

    def __init__(self, ctx: AppContext, *, folder: Path | None = None, compact: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.folder = folder or ctx.terminals.default_folder()
        self.compact = compact
        #: view → (argv, title, is a shell, extra environment): enough to start it again.
        self._meta: dict[TerminalView, tuple[list[str], str, bool, dict[str, str] | None]] = {}
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.new_button = QToolButton()
        self.new_button.setIcon(icon("plus"))
        self.new_button.setText("New terminal")
        self.new_button.setToolTip("Open a terminal (shells, Python, vim, Neovim, Docker)")
        self.new_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.new_button.setToolButtonStyle(self._button_style())
        self.new_button.clicked.connect(lambda: self.open_default())
        self.menu = QMenu(self)
        self.menu.aboutToShow.connect(self._fill_menu)
        self.new_button.setMenu(self.menu)
        row.addWidget(self.new_button)
        if not compact:
            row.addWidget(label("Folder", "KvKey"))
        self.folder_label = PathLabel(self.folder)
        row.addWidget(self.folder_label, 1)
        self.folder_button = QPushButton(icon("folder-open"), "" if compact else "Change…")
        self.folder_button.setToolTip("The folder new terminals start in")
        self.folder_button.clicked.connect(self.choose_folder)
        row.addWidget(self.folder_button)
        box.addLayout(row)
        self.status = label("", "CardCaption", wrap=True)
        self.status.hide()
        box.addWidget(self.status)

        self.stack = QStackedLayout()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._focus_current)
        self.empty = QWidget()
        empty_box = QVBoxLayout(self.empty)
        empty_box.addStretch(1)
        self.empty_hint = label("No terminal is open.", "CardCaption")
        empty_box.addWidget(self.empty_hint)
        self.empty_buttons = QHBoxLayout()
        empty_box.addLayout(self.empty_buttons)
        empty_box.addStretch(2)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.tabs)
        box.addLayout(self.stack, 1)
        self._fill_empty()

    def _button_style(self):
        from PySide6.QtCore import Qt
        return (Qt.ToolButtonStyle.ToolButtonIconOnly if self.compact
                else Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    # Profiles -------------------------------------------------------------------------------------
    def profiles(self) -> list[Profile]:
        return self.ctx.terminals.profiles()

    def _fill_menu(self) -> None:
        self.menu.clear()
        found = self.ctx.terminals.profiles(refresh=True)  # something may have been installed
        for profile in found:
            action = self.menu.addAction(icon(profile.icon), profile.title)
            action.setEnabled(profile.available)
            action.setToolTip(profile.missing or " ".join(profile.argv))
            if not profile.available:
                action.setText(f"{profile.title} — not installed")
            action.triggered.connect(lambda _=False, p=profile: self.open_profile(p))
        installable = [p for p in found if not p.available and p.install]
        if installable:
            self.menu.addSeparator()
            for profile in installable:
                action = self.menu.addAction(icon("download"), f"Install {profile.title}…")
                action.setToolTip(" ".join(profile.install))
                action.triggered.connect(lambda _=False, p=profile: self.install(p))
        self.menu.setToolTipsVisible(True)
        self.menu.addSeparator()
        self.menu.addAction(icon("square-terminal"), "A command of your own…", self.open_custom)

    def _fill_empty(self) -> None:
        while self.empty_buttons.count():
            item = self.empty_buttons.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        shown = [p for p in self.profiles() if p.available and not p.needs_choice][:5]
        for profile in shown:
            button = QPushButton(icon(profile.icon), profile.title)
            button.clicked.connect(lambda _=False, p=profile: self.open_profile(p))
            self.empty_buttons.addWidget(button)
        self.empty_buttons.addStretch(1)

    # Opening -------------------------------------------------------------------------------------------
    def open_default(self) -> TerminalView | None:
        profile = default_profile(self.profiles())
        return self.open_profile(profile) if profile else None

    def open_profile(self, profile: Profile | str, *, args: tuple[str, ...] = (),
                     folder: Path | None = None) -> TerminalView | None:
        if isinstance(profile, str):
            found = find_profile(self.profiles(), profile)
            if found is None:
                return None
            profile = found
        if not profile.available:
            self._problem(profile.missing)
            return None
        if profile.needs_choice:  # Docker
            dialog = DockerDialog(self.ctx, profile.argv[0], self)
            if not dialog.exec():
                return None
            return self.open_command(dialog.command(), dialog.title(), folder=folder, shell=True)
        return self.open_command([*profile.argv, *args], profile.title, folder=folder,
                                 shell=profile.shell)

    def install(self, profile: Profile) -> TerminalView | None:
        """Run the profile's install command in a terminal (visible, and it may ask for a password)."""
        if not profile.install:
            return None
        return self.open_command(list(profile.install), f"Install {profile.title}", shell=False)

    def open_custom(self) -> TerminalView | None:
        argv = ask_command(self, self.folder)
        return self.open_command(argv, argv[0]) if argv else None

    def open_command(self, argv: list[str], title: str, *, folder: Path | None = None,
                     shell: bool = True, env: dict[str, str] | None = None) -> TerminalView | None:
        """A new tab running ``argv`` in ``folder`` (the tabs' folder by default); ``env`` adds
        to the terminal environment."""
        folder = folder or self.folder
        try:
            session = self.ctx.terminals.start(argv, folder, title=title, env=env)
        except PtyError as exc:
            self._problem(str(exc))
            return None
        self._problem("")
        view = self._view(session)
        self._meta[view] = (list(argv), title, shell, env)
        index = self.tabs.addTab(view, icon("terminal"), self._unique(title))
        self.tabs.setTabToolTip(index, f"{' '.join(argv)}\nin {folder}")
        self.tabs.setCurrentIndex(index)
        self.stack.setCurrentWidget(self.tabs)
        view.setFocus()
        return view

    def _view(self, session: PtySession) -> TerminalView:
        view = TerminalView(session)
        view.open_path.connect(self.open_path)
        view.restart_requested.connect(lambda v=view: self._restart(v))
        session.exited.connect(lambda code, v=view: self._exited(v, code))
        session.title_changed.connect(lambda text, v=view: self._retitle(v, text))
        return view

    def _unique(self, title: str) -> str:
        names = {self.tabs.tabText(i) for i in range(self.tabs.count())}
        if title not in names:
            return title
        number = 2
        while f"{title} ({number})" in names:
            number += 1
        return f"{title} ({number})"

    def _retitle(self, view: TerminalView, text: str) -> None:
        index = self.tabs.indexOf(view)
        if index >= 0 and text:
            argv, _title, _shell, _env = self._meta.get(view, ([], "", True, None))
            self.tabs.setTabToolTip(index, f"{text}\n{' '.join(argv)}")

    def _exited(self, view: TerminalView, code: int) -> None:
        view.session.feed_local(f"\r\n\x1b[2m[{'finished' if code == 0 else f'exited with code {code}'}"
                                f" — press Enter to start it again]\x1b[0m\r\n")
        index = self.tabs.indexOf(view)
        if index >= 0:
            self.tabs.setTabIcon(index, icon("circle-stop"))

    def _restart(self, view: TerminalView) -> None:
        index = self.tabs.indexOf(view)
        argv, title, shell, env = self._meta.get(view, ([], "", True, None))
        if index < 0 or not argv:
            return
        try:
            session = self.ctx.terminals.start(argv, view.session.cwd, title=title, env=env)
        except PtyError as exc:
            view.session.feed_local(f"\r\n{exc}\r\n")
            return
        self.ctx.terminals.end(view.session)
        new = self._view(session)
        self._meta[new] = self._meta.pop(view)
        text = self.tabs.tabText(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, new, icon("terminal"), text)
        self.tabs.setCurrentIndex(index)
        self.restarted.emit(view, new)
        view.deleteLater()
        new.setFocus()

    # Closing ---------------------------------------------------------------------------------------------
    def views(self) -> list[TerminalView]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def current_view(self) -> TerminalView | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, TerminalView) else None

    def busy(self, view: TerminalView) -> bool:
        _argv, _title, shell, _env = self._meta.get(view, ([], "", True, None))
        return view.session.busy(shell)

    def unsaved_work(self) -> list[str]:
        """Tabs whose program is still busy (an editor, a command a shell is running)."""
        return [self.tabs.tabText(self.tabs.indexOf(view)) for view in self.views() if self.busy(view)]

    def close_tab(self, index: int, *, confirm: bool = True) -> bool:
        view = self.tabs.widget(index)
        if not isinstance(view, TerminalView):
            return False
        if confirm and self.busy(view):
            answer = QMessageBox.question(self, "Close terminal",
                                          f"{self.tabs.tabText(index)} is still running something. "
                                          "Close it anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.tabs.removeTab(index)
        self._meta.pop(view, None)
        self.ctx.terminals.end(view.session)
        view.deleteLater()
        if not self.tabs.count():
            self.stack.setCurrentWidget(self.empty)
        return True

    def close_all(self) -> None:
        while self.tabs.count():
            self.close_tab(0, confirm=False)

    # Folder -------------------------------------------------------------------------------------------------
    def set_folder(self, folder: Path) -> None:
        self.folder = folder
        self.folder_label.set_full_text(str(folder))

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Folder new terminals start in", str(self.folder))
        if chosen:
            self.set_folder(Path(chosen))

    def _focus_current(self, _index: int) -> None:
        view = self.current_view()
        if view is not None:
            view.setFocus()

    def _problem(self, text: str) -> None:
        """Say why a terminal did not open (no dialog: it would block the window)."""
        self.status.setText(text)
        self.status.setVisible(bool(text))
