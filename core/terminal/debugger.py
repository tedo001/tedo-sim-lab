"""Debugging a Python file with pdb, the standard library's debugger, in a terminal: the command
that starts it with the editor's breakpoints set, and where pdb has stopped, read from the
terminal screen so the editor can follow (``> /path/file.py(12)<module>()`` is pdb's own line).

Reading the screen rather than the raw output works the same with a POSIX pty and a Windows
console, which redraws lines with cursor moves instead of printing them in order.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from .screen import Screen

__all__ = ["Stop", "find_stop", "pdb_command"]

_WHERE = re.compile(r"^> (?P<path>.+?)\((?P<line>\d+)\)(?P<function>[^()\s]*)\(\)")


class Stop(NamedTuple):
    #: The screen line pdb printed it on: tells two stops at the same place apart.
    number: int
    path: Path
    line: int


def pdb_command(python: str, script: Path, args: list[str] | tuple[str, ...] = (),
                breakpoints: dict[Path, set[int] | list[int]] | None = None) -> list[str]:
    """``python -m pdb`` with ``break file:line`` for each breakpoint; with breakpoints it runs on
    to the first one, without any it stops at the script's first line."""
    command = [python, "-m", "pdb"]
    marks = {Path(path): sorted(set(lines)) for path, lines in (breakpoints or {}).items() if lines}
    for path in sorted(marks):
        command += [arg for line in marks[path] for arg in ("-c", f"break {path}:{line}")]
    if marks:
        command += ["-c", "continue"]
    return [*command, str(script), *args]


def find_stop(screen: Screen) -> Stop | None:
    """The latest place pdb reports stopping at, at or above the cursor; ``None`` if none is
    on the screen. Lines that wrapped are read whole."""
    first = screen.history_total - len(screen.scrollback)
    number = screen.history_total + screen.cy
    lowest = max(first, screen.history_total - 200)  # the screen and a little scrollback
    while number >= lowest:
        start = number
        while start > first and _wrapped(screen, start - 1):
            start -= 1
        text = "".join(screen.row_text(screen.history_line(n) or []) for n in range(start, number + 1))
        match = _WHERE.match(text.rstrip())
        if match and not match["path"].startswith("<"):
            return Stop(start, Path(match["path"].strip()), int(match["line"]))
        number = start - 1
    return None


def _wrapped(screen: Screen, number: int) -> bool:
    row = screen.history_line(number)
    return bool(row is not None and getattr(row, "wrapped", False))
