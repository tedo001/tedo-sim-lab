"""The pdb command the Code page's Debug starts, and reading where pdb stopped off the screen."""

from __future__ import annotations

from pathlib import Path

from core.terminal import Parser, Screen
from core.terminal.debugger import Stop, find_stop, pdb_command


def test_pdb_command_sets_breakpoints_then_continues(tmp_path) -> None:
    script = tmp_path / "train.py"
    helper = tmp_path / "lib" / "util.py"
    plain = pdb_command("python3", script, ["--epochs", "2"])
    assert plain == ["python3", "-m", "pdb", str(script), "--epochs", "2"]  # stops on the first line
    marked = pdb_command("python3", script, (), {script: [9, 3, 3], helper: {5}, tmp_path / "x.py": []})
    assert marked == ["python3", "-m", "pdb", "-c", f"break {helper}:5", "-c", f"break {script}:3",
                      "-c", f"break {script}:9", "-c", "continue", str(script)]


def screen_after(text: str, cols: int = 60) -> Screen:
    screen = Screen(cols, 12, 100)
    Parser(screen).feed(text)
    return screen


def test_the_latest_stop_is_found_above_the_prompt() -> None:
    first = "> /work/train.py(3)<module>()\r\n-> import numpy\r\n(Pdb) "
    screen = screen_after(first)
    assert find_stop(screen) == Stop(0, Path("/work/train.py"), 3)
    Parser(screen).feed("n\r\n> /work/train.py(4)main()\r\n-> x = 1\r\n(Pdb) l\r\n  1  \r\n  2  \r\n(Pdb) ")
    stop = find_stop(screen)
    assert (stop.path, stop.line) == (Path("/work/train.py"), 4) and stop.number == 3
    assert find_stop(screen_after("$ python train.py\r\nhello\r\n")) is None
    assert find_stop(screen_after("> <string>(1)<module>()\r\n(Pdb) ")) is None


def test_a_wrapped_location_is_read_whole_and_redraws_count_once() -> None:
    long_path = "/very/long/project/folder/with/many/levels/experiments/train_model.py"
    screen = screen_after(f"> {long_path}(12)<module>()\r\n-> run()\r\n(Pdb) ", cols=30)
    assert find_stop(screen) == Stop(0, Path(long_path), 12)
    # A Windows console redraws by moving the cursor; the same stop keeps its line number.
    redrawn = screen_after("> C:\\lab\\a.py(7)<module>()\x1b[K\r\n-> y = 2\x1b[3;1H(Pdb) ")
    assert find_stop(redrawn) == Stop(0, Path("C:\\lab\\a.py"), 7)
