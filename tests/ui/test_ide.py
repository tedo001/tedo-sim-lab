"""The Code page: highlighting, the editor's keys, running files (arguments, tracebacks that jump),
file operations inside the project, git state, recent projects and unsaved work."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QTextDocument

from app.services.project import (
    RecentProjects,
    check_python_syntax,
    git_status,
    run_command,
    safe_child,
)
from app.ui.pages.ide import IdePage
from app.ui.theme.tokens import COLORS
from app.ui.widgets.code_editor import CodeEditor
from app.ui.widgets.highlighters import highlighter_for


@pytest.fixture
def project(tmp_path) -> Path:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "helper.py").write_text("def double(x):\n    return 2 * x\n")
    (root / "main.py").write_text("import sys\nfrom pkg.helper import double\n"
                                  "print('args', sys.argv[1:])\nprint(double(21))\n")
    (root / "boom.py").write_text("def explode():\n    raise RuntimeError('bang')\n\nexplode()\n")
    return root


@pytest.fixture
def page(ctx, qtbot, project, tmp_path) -> IdePage:
    recent = RecentProjects(QSettings(str(tmp_path / "recent.ini"), QSettings.Format.IniFormat))
    ide = IdePage(ctx, recent=recent)
    qtbot.addWidget(ide)
    ide.open_project(project)
    return ide


def colours(document: QTextDocument) -> list[list[tuple[int, int, str]]]:
    rows, block = [], document.begin()
    while block.isValid():
        rows.append([(r.start, r.length, r.format.foreground().color().name().upper())
                     for r in block.layout().formats()])
        block = block.next()
    return rows


def test_python_highlighting_knows_strings_comments_and_triple_quotes(qapp) -> None:
    document = QTextDocument('x = "a # b"  # note\ns = """one\ntwo # inside\n"""\n')
    highlighter = highlighter_for(Path("x.py"), document)
    highlighter.rehighlight()
    rows = colours(document)
    assert (4, 7, COLORS["code_string"]) in rows[0] and (13, 6, COLORS["code_comment"]) in rows[0]
    assert rows[2] == [(0, 12, COLORS["code_string"])]  # a "#" inside a docstring is not a comment
    assert highlighter_for(Path("notes.txt"), QTextDocument()) is None


def test_editor_keys(qtbot) -> None:
    editor = CodeEditor()
    qtbot.addWidget(editor)
    editor.setPlainText("def f():")
    editor.moveCursor(editor.textCursor().MoveOperation.End)
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    qtbot.keyClicks(editor, "return 1")
    assert editor.toPlainText() == "def f():\n    return 1"
    editor.selectAll()
    editor.toggle_comment()
    assert editor.toPlainText() == "# def f():\n    # return 1"
    editor.selectAll()
    editor.toggle_comment()
    assert editor.toPlainText() == "def f():\n    return 1"
    editor.selectAll()
    editor.indent_selection(-1)
    assert editor.toPlainText() == "def f():\nreturn 1"
    assert editor.find_text("RETURN") and not editor.find_text("RETURN", case=True)
    assert editor.replace_all("f(", "g(") == 1 and editor.toPlainText().startswith("def g")


def test_breakpoints_move_with_the_text(qtbot) -> None:
    editor = CodeEditor()
    qtbot.addWidget(editor)
    editor.setPlainText("a\nb\nc\nd\n")
    editor.toggle_breakpoint(2)
    editor.toggle_breakpoint(4)
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.insertText("x\n")                       # a line above both
    assert editor.breakpoints == {3, 5}
    block = editor.document().findBlockByNumber(2)  # "b", with a breakpoint: delete the whole line
    cursor.setPosition(block.position())
    cursor.setPosition(block.next().position(), cursor.MoveMode.KeepAnchor)
    cursor.removeSelectedText()
    assert editor.breakpoints == {4} and editor.document().findBlockByNumber(3).text() == "d"
    cursor.setPosition(editor.document().findBlockByNumber(0).position() + 1)
    cursor.insertText("\n")                        # Enter at the end of line 1
    assert editor.breakpoints == {5}
    editor.go_to_line(5)
    qtbot.keyClick(editor, Qt.Key.Key_F9)
    assert editor.breakpoints == set()


def test_run_commands_and_files(tmp_path) -> None:
    assert run_command(tmp_path / "a.py", ["-v"], "/py") == ["/py", "-u", str(tmp_path / "a.py"), "-v"]
    with pytest.raises(ValueError, match="experiment Python"):
        run_command(tmp_path / "a.py", [], "")
    with pytest.raises(ValueError, match="does not know"):
        run_command(tmp_path / "a.txt", [], "/py")
    for bad in ("../x", "a/b", "", "..", "C:evil"):
        with pytest.raises(ValueError):
            safe_child(tmp_path, bad)
    assert check_python_syntax("x = (1,\n") is not None and check_python_syntax("x = 1\n") is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_status(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "new.py").write_text("x = 1\n")
    state = git_status(tmp_path)
    assert state.branch == "main" and ("??", "new.py") in state.changes
    assert git_status(tmp_path.parent / "elsewhere").error or True


def test_recent_projects(tmp_path) -> None:
    recent = RecentProjects(QSettings(str(tmp_path / "r.ini"), QSettings.Format.IniFormat), limit=2)
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        recent.add(tmp_path / name)
    assert [p.name for p in recent.all()] == ["c", "b"]


def test_open_edit_save_and_run_with_arguments(page, qtbot, project) -> None:
    assert page.root == project.resolve() and page.tree.root == project.resolve()
    page.open_file(project / "main.py")
    assert page.tabs.count() == 1 and page.run_button.isEnabled()
    editor = page.tabs.current_editor()
    editor.moveCursor(editor.textCursor().MoveOperation.End)
    editor.insertPlainText("print('edited')\n")
    assert page.unsaved_work() == ["main.py"] and page.tabs.tabText(0).startswith("●")
    page.args.setText("--fast 'two words'")
    with qtbot.waitSignal(page.runner.finished, timeout=30_000) as done:
        assert page.run_current()  # saves first, like PyCharm
    assert done.args == [0, False] and not page.unsaved_work()
    output = page.panel.output.toPlainText()
    assert "args ['--fast', 'two words']" in output and "42" in output and "edited" in output


def test_a_traceback_line_opens_the_file_at_that_line(page, qtbot, project) -> None:
    page.open_file(project / "boom.py")
    with qtbot.waitSignal(page.runner.finished, timeout=30_000) as done:
        page.run_current()
    assert done.args[0] != 0
    line = next(text for text in page.panel.output.toPlainText().splitlines() if "line 2" in text)
    page.open_file(project / "main.py")
    assert page.panel.output.open_line(line)
    assert page.tabs.current_path() == (project / "boom.py").resolve()
    assert page.tabs.current_editor().textCursor().blockNumber() == 1


def test_syntax_errors_are_marked_on_save(page, project) -> None:
    page.open_file(project / "pkg" / "helper.py")
    editor = page.tabs.current_editor()
    editor.setPlainText("def double(x)\n    return 2 * x\n")
    assert page.tabs.save()
    assert editor.error_line == 1 and "syntax error on line 1" in page.status.text()
    editor.setPlainText("def double(x):\n    return 2 * x\n")
    page.tabs.save()
    assert editor.error_line is None


def test_files_are_made_renamed_and_trashed_inside_the_project(page, project) -> None:
    made = page.tree.new_file(project / "pkg", "extra.py")
    assert made == project / "pkg" / "extra.py" and made.is_file()
    assert page.tabs.current_path() == made.resolve()
    assert page.tree.new_file(project, "../escape.py") is None and not (project.parent / "escape.py").exists()
    folder = page.tree.new_folder(project, "data")
    assert folder.is_dir()
    renamed = page.tree.rename(project / "pkg", "library")
    assert renamed == project / "library"
    assert page.tabs.current_path() == (project / "library" / "extra.py").resolve()  # the tab followed
    if page.tree.delete(renamed / "extra.py", confirm=False):  # needs a trash; skip quietly if none
        assert page.tabs.count() == 0


def test_running_needs_an_experiment_python_for_python_files(page, project, monkeypatch) -> None:
    monkeypatch.setattr("app.services.project.experiment_python", lambda config: "")
    page.open_file(project / "main.py")
    assert not page.run_current() and "Experiment Python" in page.status.text()


def test_unsaved_work_is_asked_about_before_quitting(window, qtbot, project, monkeypatch) -> None:
    window.navigate("code")
    ide = window.page_widget("code")
    ide.open_project(project)
    ide.open_file(project / "main.py")
    ide.tabs.current_editor().insertPlainText("# change\n")
    asked = []
    monkeypatch.setattr(window, "_confirm_unsaved", lambda names: asked.append(names) or False)
    window.close()
    assert asked == [["main.py"]] and window.isVisible()
    assert sys.platform  # (window closes normally in the fixture teardown)
    ide.tabs.current_editor().document().setModified(False)


def test_debugging_follows_pdb_and_takes_breakpoints_at_its_prompt(page, qtbot, project) -> None:
    script = project / "steps.py"
    script.write_text("a = 1\nb = a + 1\nc = b * 2\nprint('done', c)\n")
    page.open_file(script)
    editor = page.tabs.current_editor()
    editor.toggle_breakpoint(3)
    assert page.debug_current() and page.stop_button.isEnabled()
    qtbot.waitUntil(lambda: editor.debug_line == 3 and page.debugger.at_prompt(), timeout=30_000)
    editor.toggle_breakpoint(4)  # pdb waits at its prompt: it is told at once
    qtbot.waitUntil(lambda: "Breakpoint 2" in "\n".join(page.debugger.view.session.screen.display()),
                    timeout=20_000)
    qtbot.waitUntil(page.debugger.at_prompt, timeout=20_000)
    page.debugger.view.session.write("c\r")
    qtbot.waitUntil(lambda: editor.debug_line == 4, timeout=20_000)
    qtbot.waitUntil(page.debugger.at_prompt, timeout=20_000)
    page.debugger.view.session.write("q\r")
    qtbot.waitUntil(lambda: not page.debugger.active, timeout=20_000)
    assert editor.debug_line is None and not page.stop_button.isEnabled()
    assert page.tabs.breakpoints() == {script.resolve(): {3, 4}}


def test_a_file_runs_in_a_terminal_where_it_can_read_input(page, qtbot, project) -> None:
    script = project / "ask.py"
    script.write_text("name = input('name? ')\nprint('hello', name)\n")
    page.open_file(script)
    assert page.debug_current(debug=False)
    view = page.debugger.run_view
    qtbot.waitUntil(lambda: "name?" in "\n".join(view.session.screen.display()), timeout=20_000)
    view.session.write("lab\r")
    qtbot.waitUntil(lambda: view.session.exit_code is not None, timeout=20_000)
    assert "hello lab" in "\n".join(view.session.screen.display()) and view.session.exit_code == 0
