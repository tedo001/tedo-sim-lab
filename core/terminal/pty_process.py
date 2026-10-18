"""A program running in a pseudo-terminal: what the terminal widget talks to.

On Linux and macOS this is the system's own pty (standard library only); on Windows it is a
ConPTY console through pywinpty (MIT). Both read in a background thread that the caller owns
(``read`` blocks for up to ``timeout`` seconds), write without ever blocking the caller, and
can resize, tell whether the program still runs, and end it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["PtyError", "PtyProcess", "spawn"]


class PtyError(OSError):
    """The program could not be started in a terminal; the message says why."""


class PtyProcess:
    """The interface both platforms implement."""

    pid: int

    def read(self, timeout: float = 0.2) -> bytes | str | None:
        """Output so far (``b""``/``""`` when none arrived in time); ``None`` once it ended."""
        raise NotImplementedError

    def write(self, data: bytes) -> bytes:
        """Send what fits now; return what is left to send later."""
        raise NotImplementedError

    def resize(self, rows: int, cols: int) -> None:
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError

    def exit_code(self) -> int | None:
        raise NotImplementedError

    def terminate(self) -> None:
        """End the program (and, on Linux and macOS, everything it started in its terminal)."""
        raise NotImplementedError


def spawn(argv: list[str], cwd: Path, env: dict[str, str], rows: int, cols: int) -> PtyProcess:
    if sys.platform == "win32":
        return _WindowsPty(argv, cwd, env, rows, cols)
    return _PosixPty(argv, cwd, env, rows, cols)


class _PosixPty(PtyProcess):
    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], rows: int, cols: int) -> None:
        import fcntl
        import pty
        import termios

        self._fcntl, self._termios = fcntl, termios
        master, slave = pty.openpty()
        self._set_size(slave, rows, cols)

        def controlling_terminal() -> None:  # in the child, after setsid and the dup2s
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        try:
            self._process = subprocess.Popen(
                argv, stdin=slave, stdout=slave, stderr=slave, cwd=str(cwd), env=env,
                start_new_session=True, preexec_fn=controlling_terminal, close_fds=True)
        except OSError as exc:
            os.close(master)
            os.close(slave)
            raise PtyError(f"Could not start {argv[0]}: {exc.strerror or exc}") from None
        os.close(slave)
        os.set_blocking(master, False)
        self.fd = master
        self.pid = self._process.pid

    def _set_size(self, fd: int, rows: int, cols: int) -> None:
        import struct
        self._fcntl.ioctl(fd, self._termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def read(self, timeout: float = 0.2) -> bytes | None:
        import select
        if self.fd < 0:
            return None
        try:
            ready, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not ready:
            # Ended, and nothing (not even a background job) keeps the terminal open.
            return None if self._process.poll() is not None and not self._pending_output() else b""
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return b""
        except OSError:  # EIO: the terminal's other side is closed
            return None
        return data or None

    def _pending_output(self) -> bool:
        import select
        try:
            return bool(select.select([self.fd], [], [], 0)[0])
        except (OSError, ValueError):
            return False

    def write(self, data: bytes) -> bytes:
        if self.fd < 0 or not data:
            return b""
        try:
            written = os.write(self.fd, data)
        except BlockingIOError:
            return data
        except OSError:
            return b""
        return data[written:]

    def resize(self, rows: int, cols: int) -> None:
        if self.fd >= 0:
            try:
                self._set_size(self.fd, rows, cols)  # the kernel sends SIGWINCH
            except OSError:
                pass

    def alive(self) -> bool:
        return self._process.poll() is None

    def exit_code(self) -> int | None:
        code = self._process.poll()
        return code if code is None or code >= 0 else 128 - code  # signals as the shell shows them

    def terminate(self) -> None:
        if self._process.poll() is None:
            for sig, wait in ((signal.SIGHUP, 1.0), (signal.SIGKILL, 1.0)):
                try:
                    os.killpg(self.pid, sig)  # the session: the shell and its jobs
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    self._process.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            try:
                os.close(fd)
            except OSError:
                pass


class _WindowsPty(PtyProcess):
    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], rows: int, cols: int) -> None:
        import shutil

        try:
            from winpty import PTY
        except ImportError:
            raise PtyError("pywinpty is not installed; the terminal needs it on Windows.") from None
        program = shutil.which(argv[0], path=env.get("PATH")) or argv[0]
        if not Path(program).is_file():
            raise PtyError(f"Could not start {argv[0]}: it was not found on PATH.")
        self._pty = PTY(cols, rows)
        block = "\0".join(f"{key}={value}" for key, value in env.items()) + "\0"
        command_line = " " + subprocess.list2cmdline(argv[1:]) if len(argv) > 1 else None
        try:
            if command_line is None:
                self._pty.spawn(program, cwd=str(cwd), env=block)
            else:
                self._pty.spawn(program, cwd=str(cwd), env=block, cmdline=command_line)
        except Exception as exc:  # pywinpty raises its own error type
            raise PtyError(f"Could not start {argv[0]}: {exc}") from None
        self.pid = int(self._pty.pid or 0)
        self._ended_at: float | None = None

    def read(self, timeout: float = 0.2) -> str | None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                data = self._pty.read(blocking=False)
            except Exception:
                return None
            if data:
                return data
            if not self._pty.isalive():
                # Let the console flush what the program printed last.
                self._ended_at = self._ended_at or time.monotonic()
                if time.monotonic() - self._ended_at > 0.3 or self._pty.iseof():
                    return None
            if time.monotonic() >= deadline:
                return ""
            time.sleep(0.008)

    def write(self, data: bytes) -> bytes:
        try:
            self._pty.write(data.decode("utf-8", errors="replace"))
        except Exception:
            pass
        return b""

    def resize(self, rows: int, cols: int) -> None:
        try:
            self._pty.set_size(cols, rows)
        except Exception:
            pass

    def alive(self) -> bool:
        try:
            return bool(self._pty.isalive())
        except Exception:
            return False

    def exit_code(self) -> int | None:
        try:
            return self._pty.get_exitstatus()
        except Exception:
            return None

    def terminate(self) -> None:
        if self.alive() and self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)  # TerminateProcess
            except OSError:
                pass
        try:
            self._pty.cancel_io()
        except Exception:
            pass
