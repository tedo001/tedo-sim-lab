"""The Code page's engine: a project is a folder. Run a file in it (Python with the experiment
Python and the project on ``PYTHONPATH``; shell, PowerShell and batch scripts with their own
interpreters), make, rename and delete files safely inside it, read its git state, check
Python syntax, and remember recent projects (per person, in ``QSettings``)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QFile, QObject, QProcess, QProcessEnvironment, QSettings, QTimer, Signal

from core.common import AppConfig, experiment_python
from core.common.masking import mask_text
from core.common.paths import WORKSPACE_ENV

__all__ = ["ProjectRunner", "RecentProjects", "TRACEBACK_LINE", "check_python_syntax", "git_status",
           "run_command", "safe_child", "trash"]

#: ``File "C:\\p\\x.py", line 12`` in a Python traceback.
TRACEBACK_LINE = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)')
RUNNABLE = {".py": "Python", ".pyw": "Python", ".sh": "shell", ".bash": "shell", ".ps1": "PowerShell",
            ".bat": "batch", ".cmd": "batch"}


def run_command(path: Path, args: list[str], python: str) -> list[str]:
    """The command that runs ``path`` (``ValueError`` for a file the IDE cannot run)."""
    kind = RUNNABLE.get(path.suffix.lower())
    if kind == "Python":
        if not python:
            raise ValueError("No experiment Python is set up yet: Settings → Experiment Python.")
        return [python, "-u", str(path), *args]
    if kind == "shell":
        shell = shutil.which("bash") or shutil.which("sh")
        if shell is None:
            raise ValueError("No bash or sh was found to run a shell script.")
        return [shell, str(path), *args]
    if kind == "PowerShell":
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            raise ValueError("PowerShell was not found.")
        return [shell, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path), *args]
    if kind == "batch":
        if sys.platform != "win32":
            raise ValueError("Batch files run on Windows only.")
        return ["cmd", "/d", "/c", str(path), *args]
    raise ValueError(f"The IDE does not know how to run {path.suffix or 'this kind of'} files "
                     "(it runs Python, shell, PowerShell and batch scripts).")


class ProjectRunner(QObject):
    output = Signal(str, str)      # text, stream: "stdout" | "stderr" | "info"
    started = Signal(str)          # what runs
    finished = Signal(int, bool)   # exit code, stopped by the person

    def __init__(self, config: AppConfig, workspace: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config, self.workspace = config, workspace
        self._process: QProcess | None = None
        self._retired: list[QProcess] = []
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._process is not None

    def run(self, path: Path, root: Path, args: list[str] | None = None) -> None:
        """Run ``path`` with ``root`` as working folder; ``ValueError`` says why it cannot."""
        if self.running:
            raise ValueError("Something is still running: stop it first.")
        python = experiment_python(self.config)
        command = run_command(path, list(args or []), python)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", os.pathsep.join(filter(None, [str(root), env.value("PYTHONPATH", "")])))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert(WORKSPACE_ENV, str(self.workspace))
        if python:
            env.insert("PATH", str(Path(python).parent) + os.pathsep + env.value("PATH", ""))
        process = QProcess()  # no parent: kept alive from Python until the event loop is back
        self.destroyed.connect(process.kill)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(root))
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(lambda: self._read(process, "stdout"))
        process.readyReadStandardError.connect(lambda: self._read(process, "stderr"))
        process.finished.connect(lambda code, status: self._done(process, code, status))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        self._process, self._stopping = process, False
        self.started.emit(" ".join(command))
        process.start()
        process.closeWriteChannel()  # nothing can be typed into it (input() sees end of file)

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._stopping = True
        process.terminate()
        QTimer.singleShot(2_000, lambda: process.kill() if process is self._process else None)

    def _read(self, process: QProcess, stream: str) -> None:
        data = process.readAllStandardOutput() if stream == "stdout" else process.readAllStandardError()
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
        self._process = None
        self._retired.append(process)
        QTimer.singleShot(0, lambda: setattr(self, "_retired", [
            p for p in self._retired if p.state() != QProcess.ProcessState.NotRunning]))


# Files ------------------------------------------------------------------------------------------
def safe_child(folder: Path, name: str) -> Path:
    """``folder / name`` for a plain file or folder name; ``ValueError`` for anything that would
    leave ``folder`` or is not a usable name."""
    name = name.strip()
    if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\")) or ":" in name:
        raise ValueError(f"{name!r} is not a file or folder name")
    return folder / name


def trash(path: Path) -> str:
    """Move ``path`` to the system's trash (recycle bin); returns "" or why it could not."""
    result = QFile.moveToTrash(str(path))
    ok = result[0] if isinstance(result, tuple) else bool(result)
    return "" if ok else f"{path.name} could not be moved to the trash"


# Git and syntax ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GitState:
    branch: str
    #: (status letters, path relative to the repository)
    changes: list[tuple[str, str]]
    error: str = ""


def git_status(root: Path) -> GitState:
    """Branch and changed files of the repository ``root`` is in (blocking: call from a task)."""
    git = shutil.which("git")
    if git is None:
        return GitState("", [], "git is not installed")
    try:
        result = subprocess.run([git, "-C", str(root), "status", "--porcelain=v1", "--branch"],
                                capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return GitState("", [], f"git did not answer: {exc}")
    if result.returncode != 0:
        return GitState("", [], "not a git repository")
    branch, changes = "", []
    for line in result.stdout.splitlines():
        if line.startswith("## "):
            branch = line[3:].split("...")[0].removeprefix("No commits yet on ")
        elif len(line) > 3:
            changes.append((line[:2].strip() or "?", line[3:]))
    return GitState(branch, changes)


def check_python_syntax(source: str, name: str = "<file>") -> tuple[int, str] | None:
    """``(line, message)`` of the first syntax error, or ``None``."""
    try:
        compile(source, name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return (exc.lineno or 1, exc.msg or "invalid syntax")
    except ValueError as exc:  # e.g. null bytes
        return (1, str(exc))
    return None


class RecentProjects:
    """Folders opened as projects, newest first (per person, never inside a project)."""

    KEY = "code/recent_projects"

    def __init__(self, settings: QSettings | None = None, limit: int = 10) -> None:
        self.settings = settings or QSettings()
        self.limit = limit

    def all(self) -> list[Path]:
        value = self.settings.value(self.KEY, [])
        items = [value] if isinstance(value, str) else list(value or [])
        return [Path(item) for item in items if item and Path(item).is_dir()]

    def add(self, folder: Path) -> None:
        folder = folder.resolve()
        items = [folder, *(p for p in self.all() if p.resolve() != folder)][:self.limit]
        self.settings.setValue(self.KEY, [str(p) for p in items])
        self.settings.sync()
