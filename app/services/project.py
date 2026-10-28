"""The Code page's engine: a project is a folder. Run a file in it (Python with the experiment
Python, from the file's source root, as a module when it sits in a package; the source root, the
project and then the lab's code on ``PYTHONPATH``; shell, PowerShell and batch scripts with their own
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
from core.common.paths import CODE_ROOT, WORKSPACE_ENV

__all__ = ["Launch", "ProjectRunner", "RecentProjects", "TRACEBACK_LINE", "check_python_syntax", "git_status",
           "lab_packages", "project_environment", "python_launch", "run_command", "safe_child", "server_hint",
           "trash"]

#: ``File "C:\\p\\x.py", line 12`` in a Python traceback.
TRACEBACK_LINE = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)')
RUNNABLE = {".py": "Python", ".pyw": "Python", ".sh": "shell", ".bash": "shell", ".ps1": "PowerShell",
            ".bat": "batch", ".cmd": "batch"}


#: Files that mark the folder a project's imports start from.
SOURCE_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")


@dataclass(frozen=True)
class Launch:
    """How a Python file runs: what follows the interpreter (``-m package.module`` or the path),
    the working folder, and the folders its imports start from (first on ``PYTHONPATH``)."""
    target: tuple[str, ...]
    folder: Path
    paths: tuple[Path, ...] = ()


def _dotted(path: Path, source: Path) -> str | None:
    parts = [*path.relative_to(source).parent.parts, path.stem]
    return ".".join(parts) if all(part.isidentifier() for part in parts) else None


def python_launch(path: Path, root: Path) -> Launch:
    """Run a file the way its imports expect. Inside a package (folders with ``__init__.py``) it runs
    as ``-m package.module`` from the folder above the outermost package, so ``from app.config
    import x`` in ``backend/app/main.py`` finds ``backend/app``. Otherwise, below a folder holding
    ``requirements.txt``, ``pyproject.toml``, ``setup.py`` or ``setup.cfg`` (namespace packages), it
    runs from that folder. A plain script runs by path from the project folder."""
    path, root = Path(path).resolve(), Path(root).resolve()
    top = path.parent
    while (top / "__init__.py").is_file() and top != root and top.parent != top:
        top = top.parent
    source = top if top != path.parent else None
    if source is None and root in path.parents:
        for folder in path.parents:
            if folder == root:
                break
            if folder != path.parent and any((folder / name).is_file() for name in SOURCE_MARKERS):
                source = folder
                break
    if source is None:
        return Launch((str(path),), root)
    module = _dotted(path, source)
    return Launch(("-m", module) if module else (str(path),), source, (source,))


def server_hint(path: Path, launch: Launch) -> str:
    """A note for a web app module that only defines an app (FastAPI, Starlette, Flask) and starts no
    server, so running it ends at once; empty for anything else."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if "__main__" in text:
        return ""
    found = re.search(r"^(\w+)\s*=\s*(FastAPI|Starlette|Flask)\(", text, re.M)
    if not found:
        return ""
    name, kind = found.groups()
    module = launch.target[1] if launch.target[0] == "-m" else Path(path).stem
    if kind == "Flask":
        return (f"{Path(path).name} defines a Flask app but starts no server, so it ends at once. To serve "
                f"it, run in the terminal (in {launch.folder}): python -m flask --app {module}:{name} run")
    return (f"{Path(path).name} defines an ASGI app ({kind}) but starts no server, so it ends at once. To "
            f"serve it, run in the terminal (in {launch.folder}): python -m uvicorn {module}:{name} --reload")


def run_command(path: Path, args: list[str], python: str, root: Path | None = None) -> list[str]:
    """The command that runs ``path`` (``ValueError`` for a file the IDE cannot run). Python files
    run as :func:`python_launch` decides when the project folder ``root`` is given."""
    kind = RUNNABLE.get(path.suffix.lower())
    if kind == "Python":
        if not python:
            raise ValueError("No experiment Python is set up yet: Settings → Experiment Python.")
        target = python_launch(path, root).target if root is not None else (str(path),)
        return [python, "-u", *target, *args]
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


#: The lab's packages that programs run from the Code page may import (exported simulation
#: scripts use them). The rest of the lab folder, its ``app`` above all, stays off their path.
SHARED_PACKAGES = ("labs", "core")
_SHIM = '''"""Forwards to the lab's own `{name}` package (written by TEDO AI Research Lab, so that programs
run from the Code page can import it without the whole lab folder on their path)."""
import os as _os

__path__ = [{folder!r}]
_init = _os.path.join(__path__[0], "__init__.py")
with open(_init, encoding="utf-8") as _file:
    exec(compile(_file.read(), _init, "exec"))
'''


def lab_packages(workspace: Path) -> Path:
    """A folder holding forwarding packages for :data:`SHARED_PACKAGES` (rewritten only when the
    lab moved). Putting the lab folder itself on ``PYTHONPATH`` would let its regular ``app``
    package win over a project's namespace package of the same name."""
    folder = Path(workspace) / ".tedo" / "python"
    for name in SHARED_PACKAGES:
        text = _SHIM.format(name=name, folder=str(CODE_ROOT / name))
        init = folder / name / "__init__.py"
        try:
            if not init.is_file() or init.read_text(encoding="utf-8") != text:
                init.parent.mkdir(parents=True, exist_ok=True)
                init.write_text(text, encoding="utf-8")
        except OSError:
            continue
    return folder


def project_environment(root: Path, workspace: Path, python: str, base: dict[str, str],
                        sources: tuple[Path, ...] = ()) -> dict[str, str]:
    """What a project's programs run with, on top of ``base``: the file's source root and the
    project's own modules first on ``PYTHONPATH``, then the lab's ``labs`` and ``core`` packages
    (exported simulations import them) through :func:`lab_packages`, never the lab folder itself;
    unbuffered UTF-8 output; the workspace; the experiment Python's folder first on ``PATH``."""
    paths = [*map(str, sources), str(root), base.get("PYTHONPATH", ""), str(lab_packages(workspace))]
    extra = {"PYTHONPATH": os.pathsep.join(filter(None, paths)), "PYTHONUNBUFFERED": "1",
             "PYTHONIOENCODING": "utf-8", WORKSPACE_ENV: str(workspace)}
    if python:
        extra["PATH"] = str(Path(python).parent) + os.pathsep + base.get("PATH", "")
    return extra


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
        command = run_command(path, list(args or []), python, root)
        launch = python_launch(path, root) if RUNNABLE.get(path.suffix.lower()) == "Python" else Launch((),
                                                                                                        root)
        env = QProcessEnvironment.systemEnvironment()
        for key, value in project_environment(root, self.workspace, python, dict(os.environ),
                                              launch.paths).items():
            env.insert(key, value)
        process = QProcess()  # no parent: kept alive from Python until the event loop is back
        self.destroyed.connect(process.kill)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(launch.folder))
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(lambda: self._read(process, "stdout"))
        process.readyReadStandardError.connect(lambda: self._read(process, "stderr"))
        process.finished.connect(lambda code, status: self._done(process, code, status))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        self._process, self._stopping = process, False
        self.started.emit(" ".join(command))
        if launch.paths:
            self.output.emit(f"Running from {launch.folder} (its imports start there)\n", "info")
        hint = server_hint(path, launch) if RUNNABLE.get(path.suffix.lower()) == "Python" else ""
        if hint:
            self.output.emit(hint + "\n", "info")
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
