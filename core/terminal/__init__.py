"""Terminals: a VT100/xterm screen and parser, programs in a pseudo-terminal, and the profiles
(shells, editors, Docker) a terminal can start. No Qt here; the app paints the screen."""

from .parser import Parser
from .profiles import Profile, default_profile, find_profile, profiles, terminal_environment
from .pty_process import PtyError, PtyProcess, spawn
from .screen import Screen

__all__ = ["Parser", "Profile", "PtyError", "PtyProcess", "Screen", "default_profile",
           "find_profile", "profiles", "spawn", "terminal_environment"]
