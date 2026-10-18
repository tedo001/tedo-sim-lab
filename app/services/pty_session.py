"""A terminal session: a program in a pseudo-terminal, its screen, and the thread that reads it.

The reading thread only queues what arrives; the screen is changed on the GUI thread (so
painting never races the parser), which also writes the program's queries' replies back and
sends what was typed without ever blocking (what does not fit now is retried shortly).
"""

from __future__ import annotations

import codecs
import threading
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from core.common.masking import mask_text
from core.terminal import Parser, PtyError, Screen, spawn

__all__ = ["PtyError", "PtySession"]


class PtySession(QObject):
    changed = Signal()          # the screen changed: repaint
    title_changed = Signal(str)
    exited = Signal(int)        # exit code (-1 unknown)
    _arrived = Signal()         # from the reading thread (queued to the GUI thread)

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], *, rows: int = 24,
                 cols: int = 80, scrollback: int = 5000, title: str = "",
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.argv, self.cwd, self.env = list(argv), Path(cwd), env
        self.screen = Screen(cols, rows, scrollback)
        self.parser = Parser(self.screen)
        self.title = title or Path(argv[0]).stem
        self.exit_code: int | None = None
        self._process = None
        self._queue: deque = deque()
        self._notified = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._outbox = b""
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(15)
        self._flush_timer.timeout.connect(self._flush)
        self._arrived.connect(self._drain, Qt.ConnectionType.QueuedConnection)

    # Life -----------------------------------------------------------------------------------------
    def start(self) -> None:
        """Start the program; ``PtyError`` says why it could not."""
        self._process = spawn(self.argv, self.cwd, self.env, self.screen.rows, self.screen.cols)
        self._thread = threading.Thread(target=self._read_loop, name=f"pty-{self.title}", daemon=True)
        self._thread.start()

    @property
    def pid(self) -> int:
        return self._process.pid if self._process is not None else 0

    @property
    def alive(self) -> bool:
        return self._process is not None and self.exit_code is None and self._process.alive()

    def busy(self, shell: bool = True) -> bool:
        """Something would be lost by closing: a program runs (for a shell: a child process)."""
        if not self.alive:
            return False
        if not shell:
            return True
        try:
            import psutil
            return bool(psutil.Process(self.pid).children())
        except Exception:
            return False

    def terminate(self) -> None:
        """End the program and stop reading (idempotent; used on close)."""
        self._stop.set()
        self._flush_timer.stop()
        process, self._process = self._process, None
        if process is not None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None
        self._process = process  # keep it for exit_code; it no longer runs

    # Reading ------------------------------------------------------------------------------------------
    def _read_loop(self) -> None:
        process = self._process
        while not self._stop.is_set():
            data = process.read(0.2)
            if data is None:
                break
            if data:
                self._queue.append(data)
                self._notify()
        self._queue.append(None)
        self._notify()

    def _notify(self) -> None:
        if not self._notified:
            self._notified = True
            self._arrived.emit()

    def _drain(self) -> None:
        self._notified = False
        ended, chunks = False, []
        while self._queue:
            item = self._queue.popleft()
            if item is None:
                ended = True
                break
            chunks.append(item if isinstance(item, str) else self._decoder.decode(item))
        if chunks:
            title = self.screen.title
            self.parser.feed(mask_text("".join(chunks)))
            if self.screen.responses:
                replies, self.screen.responses = self.screen.responses, []
                self.write("".join(replies))
            if self.screen.title != title:
                self.title_changed.emit(self.screen.title)
            self.changed.emit()
        if ended and self.exit_code is None:
            self._ended()

    def _ended(self) -> None:
        process = self._process
        code = None
        if process is not None:
            for _ in range(20):  # the exit status can lag the end of output a little
                code = process.exit_code()
                if code is not None:
                    break
                threading.Event().wait(0.01)
        self.exit_code = -1 if code is None else code
        self._flush_timer.stop()
        self.exited.emit(self.exit_code)

    def feed_local(self, text: str) -> None:
        """Show ``text`` on the screen without sending it to the program (status lines)."""
        self.parser.feed(text)
        self.changed.emit()

    # Writing ------------------------------------------------------------------------------------------
    def write(self, data: str | bytes) -> None:
        if self._process is None or self.exit_code is not None:
            return
        self._outbox += data.encode("utf-8") if isinstance(data, str) else data
        self._flush()

    def paste(self, text: str) -> None:
        """Paste ``text``; bracketed when the program asked for it (so editors do not auto-indent
        it and shells do not run it line by line)."""
        text = text.replace("\r\n", "\r").replace("\n", "\r")
        if self.screen.bracketed_paste:
            text = "\x1b[200~" + text.replace("\x1b[201~", "") + "\x1b[201~"
        self.write(text)

    def _flush(self) -> None:
        if not self._outbox or self._process is None:
            self._flush_timer.stop()
            return
        self._outbox = self._process.write(self._outbox)
        if self._outbox:
            self._flush_timer.start()
        else:
            self._flush_timer.stop()

    def resize(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.screen.rows, self.screen.cols):
            return
        self.screen.resize(rows, cols)
        if self._process is not None and self.exit_code is None:
            self._process.resize(self.screen.rows, self.screen.cols)
        self.changed.emit()
