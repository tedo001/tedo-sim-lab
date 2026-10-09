"""The Terminal page's engine: one shell process per command, in a working folder that ``cd``
changes. It is a command runner, not a terminal emulator: no input is sent to a running command
(its standard input is closed), and full-screen programs (vim, top) do not work.

The shell is PowerShell on Windows and ``$SHELL`` (or bash, then sh) elsewhere, unless
``terminal_shell`` in settings.yaml names one. Commands see the workspace as
``TEDO_LAB_WORKSPACE`` and the experiment Python first on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from core.common import AppConfig, AppPaths, experiment_python
from core.common.masking import mask_text
from core.common.paths import WORKSPACE_ENV

__all__ = ["ShellSession", "shell_command"]


def shell_command(config: AppConfig, command: str, *, windows: bool | None = None) -> tuple[str, list[str]]:
    """(program, arguments) that run ``command`` in the configured shell."""
    windows = sys.platform == "win32" if windows is None else windows
    shell = config.terminal_shell.strip()
    if shell in ("", "auto"):
        if windows:
            shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        else:
            shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
    name = Path(shell).name.lower()
    if name.startswith(("powershell", "pwsh")):
        return shell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
    if name.startswith("cmd"):
        return shell, ["/d", "/c", command]
    return shell, ["-c", command]


class ShellSession(QObject):
    output = Signal(str, str)      # text, stream ("stdout", "stderr", "info")
    started = Signal(str)          # command
    finished = Signal(int, bool)   # exit code, stopped by the person

    def __init__(self, paths: AppPaths, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config = paths, config
        self.cwd = paths.workspace
        self.history: list[str] = []
        self._process: QProcess | None = None
        self._retired: list[QProcess] = []
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._process is not None

    def environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert(WORKSPACE_ENV, str(self.paths.workspace))
        python_dir = str(Path(experiment_python(self.config)).parent)
        env.insert("PATH", python_dir + os.pathsep + env.value("PATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        return env

    def run(self, command: str) -> bool:
        """Run ``command``; ``False`` when another one is still running. ``cd`` is handled here."""
        command = command.strip()
        if not command or self.running:
            return False
        if not self.history or self.history[-1] != command:
            self.history.append(command)
        if command in ("cd", "cd ~") or command.startswith("cd "):
            self.started.emit(command)
            self._cd(command[2:].strip())
            self.finished.emit(0, False)
            return True
        program, arguments = shell_command(self.config, command)
        process = QProcess()  # no parent: kept alive from Python until the event loop is back
        self.destroyed.connect(process.kill)  # a page that goes away takes its command with it
        process.setProgram(program)
        process.setArguments(arguments)
        process.setWorkingDirectory(str(self.cwd))
        process.setProcessEnvironment(self.environment())
        process.readyReadStandardOutput.connect(lambda: self._read(process, "stdout"))
        process.readyReadStandardError.connect(lambda: self._read(process, "stderr"))
        process.finished.connect(lambda code, status: self._done(process, code, status))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        self._process, self._stopping = process, False
        self.started.emit(command)
        process.start()
        process.closeWriteChannel()  # nothing is typed into a running command
        return True

    def stop(self) -> None:
        """Ask the command to end, then kill it after two seconds."""
        if self._process is None:
            return
        self._stopping = True
        process = self._process
        process.terminate()
        QTimer.singleShot(2_000, lambda: process.kill() if process is self._process else None)

    def _cd(self, target: str) -> None:
        target = target.strip().strip('"').strip("'")
        path = Path(os.path.expanduser(target or "~"))
        path = path if path.is_absolute() else self.cwd / path
        try:
            path = path.resolve(strict=True)
        except OSError:
            self.output.emit(f"cd: no such folder: {target}\n", "stderr")
            return
        if not path.is_dir():
            self.output.emit(f"cd: not a folder: {target}\n", "stderr")
            return
        self.cwd = path

    def _read(self, process: QProcess, stream: str) -> None:
        data = (process.readAllStandardOutput() if stream == "stdout" else process.readAllStandardError())
        text = bytes(data.data()).decode("utf-8", errors="replace")
        if text:
            self.output.emit(mask_text(text), stream)

    def _done(self, process: QProcess, code: int, status: QProcess.ExitStatus) -> None:
        if process is not self._process:
            return
        self._read(process, "stdout")
        self._read(process, "stderr")
        stopped = self._stopping
        self._release(process)
        self.finished.emit(code if status == QProcess.ExitStatus.NormalExit else -1, stopped)

    def _error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and process is self._process:
            self.output.emit(f"Could not start {process.program()}: {process.errorString()}\n", "stderr")
            self._release(process)
            self.finished.emit(-1, False)

    def _release(self, process: QProcess) -> None:
        # Still inside the process's own signal: keep it alive until the event loop is back.
        self._process = None
        self._retired.append(process)
        QTimer.singleShot(0, self._drop_retired)

    def _drop_retired(self) -> None:
        self._retired = [p for p in self._retired if p.state() != QProcess.ProcessState.NotRunning]

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.waitForFinished(1_000)
