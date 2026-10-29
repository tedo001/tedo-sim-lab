"""Standard input of a worker process that reads requests or a cancel channel on a thread.

On Windows a thread blocked reading a pipe makes every child process that would inherit that pipe
hang at start (``git`` for the snapshot, the tools libraries launch while importing or loading a
model), so a worker hands its stdin to a private, non-inheritable copy and points standard input at
the null device before it starts anything. Never undo this."""

from __future__ import annotations

import os
import sys
from typing import TextIO

__all__ = ["detach_stdin"]


def detach_stdin() -> TextIO:
    """Hand stdin to a private file and point standard input at the null device; returns the stream
    the reading thread should use (``sys.stdin`` itself when it cannot be detached)."""
    try:
        private = os.dup(sys.stdin.fileno())  # not inheritable (PEP 446)
    except (OSError, ValueError, AttributeError):
        return sys.stdin
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":  # make sure child processes see the null device as their stdin
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(0))  # STD_INPUT_HANDLE
    sys.stdin = open(os.devnull, encoding="utf-8")  # lives as long as the process
    return open(private, encoding="utf-8", errors="replace")
