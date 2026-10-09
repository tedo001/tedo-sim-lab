"""A local web interface the lab starts as a process (the MLflow UI, Jupyter Lab), bound to
127.0.0.1 on the first free port, reported "running" once the port answers, and stopped with
the app."""

from __future__ import annotations

import socket

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from core.common import AppConfig, AppPaths, experiment_python

__all__ = ["LocalWebUi", "free_port"]


def free_port(start: int = 5000, tries: int = 50) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"no free port between {start} and {start + tries - 1}")


class LocalWebUi(QObject):
    #: First port tried.
    START_PORT = 5000

    state_changed = Signal(str)   # "stopped" | "starting" | "running" | "failed"

    def __init__(self, paths: AppPaths, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config = paths, config
        self.process: QProcess | None = None
        self._retired: list[QProcess] = []
        self.port: int | None = None
        self.state = "stopped"
        self.output = ""

    @property
    def url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port and self.state == "running" else None

    def command(self, port: int) -> list[str]:
        """Arguments for the experiment Python."""
        raise NotImplementedError

    def prepare(self) -> None:
        """Called before starting (create folders)."""

    def start(self) -> None:
        if self.process is not None:
            return
        self.prepare()
        self.port = free_port(self.START_PORT)
        process = QProcess()
        process.setProgram(experiment_python(self.config))
        process.setArguments(self.command(self.port))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        self.output = ""
        self._set("starting")
        process.start()
        QTimer.singleShot(500, self._poll)

    def _read(self) -> None:
        if self.process is not None:
            self.output = (self.output + bytes(self.process.readAllStandardOutput().data()).decode(
                "utf-8", errors="replace"))[-4000:]

    def _poll(self, attempt: int = 0) -> None:
        if self.process is None or self.state != "starting":
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", self.port or 0)) == 0:
                self._set("running")
                return
        if attempt < 120:  # up to a minute: the first start creates the database tables
            QTimer.singleShot(500, lambda: self._poll(attempt + 1))
        else:
            self.stop()
            self._set("failed")

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self.process is not None:
            self.output = f"could not start {self.process.program()}: {self.process.errorString()}"
            self._release()
            self._set("failed")

    def _finished(self, *_: object) -> None:
        self._read()
        failed = self.state == "starting"
        self._release()
        self._set("failed" if failed else "stopped")

    def _release(self) -> None:
        """Forget the process while it is still emitting: keep it alive until the event loop is
        back (deleting a QProcess inside its own signal corrupts the heap under PySide6)."""
        if self.process is not None:
            self._retired.append(self.process)
            self.process = None
            QTimer.singleShot(0, lambda: self._retired.clear())

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            process.finished.disconnect(self._finished)
            process.errorOccurred.disconnect(self._error)
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
        self._set("stopped")

    def _set(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)
