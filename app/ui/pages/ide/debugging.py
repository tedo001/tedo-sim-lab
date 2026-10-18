"""Debug and Run in terminal for the Code page. Both use a terminal tab in the bottom panel, so a
program can read what you type. Debug runs the file under pdb, the standard library's debugger,
with the editor's breakpoints set; the editor follows every line pdb stops at. Breakpoints set
or cleared while pdb waits at its prompt reach it at once; otherwise from the next Debug."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.common import experiment_python
from core.terminal.debugger import Stop, find_stop, pdb_command

from ....services.context import AppContext
from ....services.project import project_environment, run_command
from ...widgets.terminal_view import TerminalView
from .console import BottomPanel
from .editors import EditorTabs

__all__ = ["Debugger"]

PROMPT = "(Pdb) "


class Debugger(QObject):
    message = Signal(str)
    #: Debugging started (True) or ended (False).
    active_changed = Signal(bool)

    def __init__(self, ctx: AppContext, tabs: EditorTabs, panel: BottomPanel,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.ctx, self.tabs, self.panel = ctx, tabs, panel
        self.view: TerminalView | None = None      # the Debug tab
        self.run_view: TerminalView | None = None  # the Run in terminal tab
        self.last_stop: Stop | None = None
        self._active = False  # from start until the Debug tab's program has ended
        panel.terminals.restarted.connect(self._restarted)
        tabs.breakpoint_toggled.connect(self._toggled)

    @property
    def active(self) -> bool:
        return self._active

    def start(self, path: Path, root: Path, args: list[str], *, debug: bool = True) -> bool:
        """Debug ``path`` (or just run it, ``debug=False``) in a terminal tab in ``root``,
        replacing the previous one of the same kind."""
        python = experiment_python(self.ctx.config)
        try:
            if debug:
                if path.suffix.lower() not in (".py", ".pyw"):
                    raise ValueError("Debug runs Python files; open a .py file.")
                if not python:
                    raise ValueError("No experiment Python is set up yet: Settings → Experiment Python.")
                argv = pdb_command(python, path, args, self.tabs.breakpoints())
            else:
                argv = run_command(path, args, python)
        except ValueError as exc:
            self.message.emit(str(exc))
            return False
        env = project_environment(root, self.ctx.paths.workspace, python, self.ctx.terminals.environment())
        self._close(self.view if debug else self.run_view)
        title = f"{'Debug' if debug else 'Run'} {path.name}"
        view = self.panel.terminals.open_command(argv, title, folder=root, shell=False, env=env)
        if view is None:
            self.message.emit(self.panel.terminals.status.text() or f"Could not start {argv[0]}")
            return False
        self.panel.show_terminal()
        if debug:
            self._follow(view)
            marks = sum(len(lines) for lines in self.tabs.breakpoints().values())
            self.message.emit(f"Debugging {path.name}: " + (
                f"runs to the first of {marks} breakpoint{'s' if marks != 1 else ''}" if marks else
                "stopped on the first line") + " · n next, s step in, c continue, p name print, q quit")
        else:
            self.run_view = view
        return True

    def stop(self) -> None:
        self._close(self.view)

    # Following pdb ---------------------------------------------------------------------------------
    def _follow(self, view: TerminalView) -> None:
        self.view, self.last_stop, self._active = view, None, True
        view.session.changed.connect(lambda v=view: self._changed(v))
        view.session.exited.connect(lambda _code, v=view: self._ended(v))
        self.active_changed.emit(True)

    def _changed(self, view: TerminalView) -> None:
        if view is not self.view:
            return
        stop = find_stop(view.session.screen)
        if stop is None or stop == self.last_stop:
            return
        self.last_stop = stop
        path = stop.path if stop.path.is_absolute() else view.session.cwd / stop.path
        if path.is_file():
            self.tabs.show_debug_line(path, stop.line)

    def _ended(self, view: TerminalView) -> None:
        if view is self.view:
            self._active = False
            self.tabs.show_debug_line(None)
            self.active_changed.emit(False)

    def _restarted(self, old: TerminalView, new: TerminalView) -> None:
        if old is self.view:
            self._follow(new)
        elif old is self.run_view:
            self.run_view = new

    def at_prompt(self) -> bool:
        """pdb waits for a command, and nothing has been typed yet."""
        if not self.active:
            return False
        screen = self.view.session.screen
        return screen.row_text(screen.lines[screen.cy]).rstrip() + " " == PROMPT

    def _toggled(self, path: Path, line: int, on: bool) -> None:
        """Tell a waiting pdb about a breakpoint at once."""
        if not self.active:
            return
        if not self.at_prompt():
            self.message.emit("pdb is busy: the breakpoint applies from the next Debug.")
            return
        self.view.session.write(f"{'break' if on else 'clear'} {path}:{line}\r")

    def _close(self, view: TerminalView | None) -> None:
        if view is None:
            return
        index = self.panel.terminals.tabs.indexOf(view)
        if index >= 0:
            self.panel.terminals.close_tab(index, confirm=False)
        if view is self.view:
            self.view, self._active = None, False
            self.tabs.show_debug_line(None)
            self.active_changed.emit(False)
        elif view is self.run_view:
            self.run_view = None
