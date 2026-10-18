"""Real terminals: a shell runs commands, programs end and restart, busy tabs are noticed, vim
edits a file, and the widget's keys, selection, paste and mouse reporting behave like xterm."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent

from app.ui.pages.terminal_tabs import TerminalTabs
from app.ui.widgets.terminal_keys import encode
from core.terminal import Screen

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell commands")


@pytest.fixture
def tabs(ctx, qtbot) -> TerminalTabs:
    widget = TerminalTabs(ctx)
    widget.resize(900, 500)
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    widget.close_all()


def lines(view) -> list[str]:
    return [line.rstrip() for line in view.session.screen.display()]


def shown(view, text: str) -> bool:
    return any(line.strip() == text for line in lines(view))


def test_a_shell_runs_commands(tabs, qtbot) -> None:
    view = tabs.open_default()
    assert view is not None and view.session.alive
    view.session.write("echo tedo-marker\r")
    qtbot.waitUntil(lambda: shown(view, "tedo-marker"), timeout=20_000)
    assert tabs.tabs.count() == 1 and tabs.stack.currentWidget() is tabs.tabs


@POSIX
def test_the_program_sees_the_terminal_size_and_the_workspace(tabs, qtbot, ctx) -> None:
    view = tabs.open_profile("bash")

    def settled() -> bool:  # the layout has given the view its size and the program knows it
        screen = view.session.screen
        return not view._resize_timer.isActive() and (screen.rows, screen.cols) == view.grid_size()

    qtbot.waitUntil(settled, timeout=5_000)
    qtbot.wait(100)
    assert settled()
    rows, cols = view.session.screen.rows, view.session.screen.cols
    view.session.write('tput cols; tput lines; echo "ws=$TEDO_LAB_WORKSPACE"; echo "$TERM"\r')
    qtbot.waitUntil(lambda: shown(view, "xterm-256color"), timeout=20_000)
    assert shown(view, str(cols)) and shown(view, str(rows)) and shown(view, f"ws={ctx.paths.workspace}")


def test_a_program_that_ends_can_be_started_again(tabs, qtbot) -> None:
    view = tabs.open_default()
    with qtbot.waitSignal(view.session.exited, timeout=20_000) as ended:
        view.session.write("exit 3\r")
    assert ended.args == [3]
    qtbot.waitUntil(lambda: any("exited with code 3" in line for line in lines(view)), timeout=5_000)
    view.setFocus()
    qtbot.keyClick(view, Qt.Key.Key_Return)
    restarted = tabs.current_view()
    assert restarted is not view and restarted.session.alive and tabs.tabs.count() == 1


@POSIX
def test_busy_tabs_are_reported_and_closing_ends_them(tabs, qtbot) -> None:
    view = tabs.open_default()
    # A shell starting up runs its rc files (short-lived children); once settled it is not busy.
    qtbot.waitUntil(lambda: tabs.unsaved_work() == [], timeout=20_000)
    view.session.write("sleep 30\r")
    qtbot.waitUntil(lambda: tabs.unsaved_work() == [tabs.tabs.tabText(0)], timeout=20_000)
    session = view.session
    assert tabs.close_tab(0, confirm=False)
    assert tabs.tabs.count() == 0 and tabs.stack.currentWidget() is tabs.empty
    qtbot.waitUntil(lambda: not session.alive, timeout=10_000)  # the shell and its job end


@pytest.mark.skipif(shutil.which("vim") is None, reason="vim is not installed")
def test_vim_edits_a_file(tabs, qtbot, tmp_path) -> None:
    note = tmp_path / "note.txt"
    view = tabs.open_profile("vim", args=("-u", "NONE", str(note)))
    qtbot.waitUntil(lambda: view.session.screen.in_alt_screen, timeout=10_000)
    assert tabs.unsaved_work() == ["Vim"]  # an editor is always worth asking about
    view.session.write("ihello from vim\x1b:wq\r")
    qtbot.waitUntil(lambda: view.session.exit_code is not None, timeout=10_000)
    assert note.read_text() == "hello from vim\n" and not view.session.screen.in_alt_screen


def test_missing_programs_are_explained_not_started(tabs) -> None:
    from core.terminal import Profile
    assert tabs.open_profile(Profile("nope", "Nope", missing="Nope is not installed")) is None
    assert tabs.status.text() == "Nope is not installed" and tabs.tabs.count() == 0
    assert tabs.open_command(["/no/such/program-xyz"], "broken") is None
    assert "program-xyz" in tabs.status.text()


def key(code, modifiers=Qt.KeyboardModifier.NoModifier, text: str = "") -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, code, modifiers, text)


def test_keys_become_xterm_bytes(qapp) -> None:
    screen = Screen()
    ctrl, alt, shift = (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier,
                        Qt.KeyboardModifier.ShiftModifier)
    assert encode(key(Qt.Key.Key_Up), screen) == "\x1b[A"
    assert encode(key(Qt.Key.Key_Up, ctrl), screen) == "\x1b[1;5A"
    assert encode(key(Qt.Key.Key_F5), screen) == "\x1b[15~"
    assert encode(key(Qt.Key.Key_F1), screen) == "\x1bOP"
    assert encode(key(Qt.Key.Key_PageUp, shift | ctrl), screen) == "\x1b[5;6~"
    assert encode(key(Qt.Key.Key_C, ctrl, "\x03"), screen) == "\x03"
    assert encode(key(Qt.Key.Key_BracketLeft, ctrl), screen) == "\x1b"
    assert encode(key(Qt.Key.Key_X, alt, "x"), screen) == "\x1bx"
    assert encode(key(Qt.Key.Key_At, ctrl | alt, "@"), screen) == "@"  # AltGr on Windows
    assert encode(key(Qt.Key.Key_Backtab, shift), screen) == "\x1b[Z"
    assert encode(key(Qt.Key.Key_Return, text="\r"), screen) == "\r"
    assert encode(key(Qt.Key.Key_Backspace), screen) == "\x7f"
    assert encode(key(Qt.Key.Key_Shift, shift), screen) == ""
    screen.app_cursor = True
    assert encode(key(Qt.Key.Key_Left), screen) == "\x1bOD"


@pytest.fixture
def idle_view(ctx, qtbot):
    """A terminal view whose session never started: what it sends is recorded instead."""
    from app.services.pty_session import PtySession
    from app.ui.widgets.terminal_view import TerminalView

    session = PtySession([sys.executable], ctx.paths.workspace, {}, rows=6, cols=30)
    sent: list[str] = []
    session.write = lambda data: sent.append(data if isinstance(data, str) else data.decode())
    view = TerminalView(session)
    view.resize(400, 200)
    qtbot.addWidget(view)
    view.show()
    return view, sent


def cell_point(view, col: int, row: int) -> QPoint:
    return QPoint(int((col + 0.5) * view.fonts.width), int((row + 0.5) * view.fonts.height))


def test_selection_copies_text_and_paste_is_bracketed(idle_view, qtbot) -> None:
    view, sent = idle_view
    view.session.feed_local("first line\r\nsecond words here")
    first = view.first_visible_line()
    view.selection = ((first, 0), (first + 1, 6))
    assert view.selected_text() == "first line\nsecond"
    qtbot.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=cell_point(view, 9, 1))
    assert view.selected_text() == "words"
    view.copy()
    assert QGuiApplication.clipboard().text() == "words"
    view.session.feed_local("\x1b[?2004h")
    QGuiApplication.clipboard().setText("a\nb")
    view.paste()
    assert sent[-1] == "\x1b[200~a\rb\x1b[201~"


def test_mouse_reports_go_to_programs_that_ask(idle_view, qtbot) -> None:
    view, sent = idle_view
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=cell_point(view, 2, 1))
    assert sent == []  # no mouse mode: a local click
    view.session.feed_local("\x1b[?1000h\x1b[?1006h")
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=cell_point(view, 2, 1))
    assert sent[-2:] == ["\x1b[<0;3;2M", "\x1b[<0;3;2m"]
    count = len(sent)
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=cell_point(view, 2, 1))
    assert len(sent) == count  # Shift keeps the mouse local


def test_ctrl_click_opens_a_file_from_a_traceback(idle_view, qtbot, tmp_path) -> None:
    view, _sent = idle_view
    script = tmp_path / "boom.py"
    script.write_text("x = 1\n")
    view.session.feed_local(f'  File "{script}", line 1\r\n')
    row = next(i for i, line in enumerate(lines(view)) if "File" in line)
    with qtbot.waitSignal(view.open_path, timeout=2_000) as opened:
        assert view.open_link_at(cell_point(view, 4, row))
    assert opened.args == [script, 1]


def test_window_shortcuts_go_to_the_program_except_the_panel_key(window, qtbot) -> None:
    window.set_terminal_open(True)
    view = window.terminal_panel.terminals.current_view()
    assert view is not None and window.terminal_open() and window.ui_state.terminal_open
    sent: list[str] = []
    view.session.write = lambda data: sent.append(data)
    view.setFocus()
    qtbot.waitUntil(view.hasFocus, timeout=2_000)
    qtbot.keyClick(view, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert sent == ["\x02"] and not window.sidebar_collapsed()  # Ctrl+B reached the program
    window.toggle_terminal()
    assert not window.terminal_open() and not window.ui_state.terminal_open


def test_the_terminal_page_opens_the_default_shell(window, qtbot) -> None:
    window.navigate("terminal")
    page = window.page_widget("terminal")
    qtbot.waitUntil(lambda: len(page.terminals.views()) == 1, timeout=10_000)
    assert page.terminals.current_view().session.cwd == window.ctx.paths.workspace


def test_the_code_page_terminal_starts_in_the_project(ctx, qtbot, tmp_path) -> None:
    from PySide6.QtCore import QSettings

    from app.services.project import RecentProjects
    from app.ui.pages.ide import IdePage

    project = tmp_path / "project"
    project.mkdir()
    page = IdePage(ctx, recent=RecentProjects(QSettings(str(tmp_path / "r.ini"), QSettings.Format.IniFormat)))
    qtbot.addWidget(page)
    page.open_project(project)
    assert ctx.terminals.project_root == project.resolve()
    page.panel.show_terminal()
    qtbot.waitUntil(lambda: page.panel.terminals.current_view() is not None, timeout=10_000)
    assert page.panel.terminals.current_view().session.cwd == project.resolve()


def test_closing_the_app_ends_every_terminal(ctx, tabs, qtbot) -> None:
    view = tabs.open_default()
    session = view.session
    ctx.terminals.shutdown()
    assert not session.alive and ctx.terminals.sessions == []
    assert isinstance(Path(session.argv[0]), Path)
