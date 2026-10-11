"""The *experiment Python*: the environment that runs workers (training, evaluation,
benchmarks), the hardware probe and plugin installs.

From a source checkout it is simply the app's own interpreter. The installed app (a
cx_Freeze build) cannot pip-install into itself and does not ship PyTorch, so it uses a
separate environment: one a person points it at in Settings, or one the lab creates on request
in a per-user folder (never the install folder) from a Python 3.11+ found on the machine, with
PyTorch (the CUDA build when an NVIDIA GPU is present) and ``configs/experiment-requirements.txt``.

Everything here is blocking (subprocesses); the app calls it from background tasks.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CODE_ROOT

__all__ = ["MINIMUM", "PythonFacts", "Step", "default_env_dir", "describe_python", "env_python",
           "find_base_python", "frozen", "has_nvidia_gpu", "python_version", "requirements_file",
           "setup_steps", "torch_build"]

MINIMUM = (3, 11)
#: PyTorch's own package indexes; CUDA 12.6 wheels run on any NVIDIA driver made for CUDA 12.
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu126",
               "cpu": "https://download.pytorch.org/whl/cpu"}
#: What :func:`describe_python` checks, as (import name, what it is for).
NEEDED = (("torch", "training image models"), ("sklearn", "the Classical ML lab"),
          ("yaml", "reading experiment.yaml"), ("pydantic", "reading experiment.yaml"),
          ("mlflow", "MLflow tracking"))


def frozen() -> bool:
    """True inside the installed (cx_Freeze) app."""
    return bool(getattr(sys, "frozen", False))


def requirements_file(code_root: Path = CODE_ROOT) -> Path:
    return code_root / "configs" / "experiment-requirements.txt"


def default_env_dir() -> Path:
    """Per-user, outside any workspace and outside the install folder."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "TEDO AI Lab" / "experiment-python"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "tedo-ai-lab" / "experiment-python"


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def python_version(command: list[str]) -> tuple[int, int] | None:
    """(major, minor) of the Python ``command`` starts, or ``None`` when it does not start."""
    result = _run([*command, "-c", "import sys; print(*sys.version_info[:2])"], 60)
    parts = result.stdout.split() if result else []
    return (int(parts[0]), int(parts[1])) if result and result.returncode == 0 and len(parts) == 2 else None


def find_base_python() -> list[str] | None:
    """A Python 3.11+ installed on this machine to make the environment from (the Windows
    ``py`` launcher first, then ``python3.x`` / ``python`` on PATH); ``None`` when there is none."""
    candidates: list[list[str]] = []
    if os.name == "nt" and shutil.which("py"):
        candidates += [["py", f"-3.{minor}"] for minor in (12, 13, 11, 14)]
    for name in ("python3.12", "python3.13", "python3.11", "python3", "python"):
        found = shutil.which(name)
        if found and not (frozen() and Path(found).resolve() == Path(sys.executable).resolve()):
            candidates.append([found])
    for command in candidates:
        version = python_version(command)
        if version and version >= MINIMUM:
            return command
    return None


def has_nvidia_gpu() -> bool:
    """``nvidia-smi`` exists and lists a GPU (asked without PyTorch)."""
    tool = shutil.which("nvidia-smi")
    result = _run([tool, "-L"], 20) if tool else None
    return bool(result and result.returncode == 0 and "GPU" in result.stdout)


def torch_build(*, cpu_only: bool = False, gpu: bool | None = None, system: str | None = None) -> str:
    """``cuda``, ``cpu``, or ``default`` (macOS: the standard wheels already use Apple's GPU)."""
    if (system or platform.system()) == "Darwin":
        return "default"
    if cpu_only:
        return "cpu"
    return "cuda" if (has_nvidia_gpu() if gpu is None else gpu) else "cpu"


@dataclass(frozen=True)
class Step:
    title: str
    command: list[str]
    #: Tried when ``command`` fails (PyTorch from PyPI when its own index cannot be reached).
    fallback: list[str] | None = None


def setup_steps(env_dir: Path, base_python: list[str], build: str,
                requirements: Path | None = None) -> list[Step]:
    """The commands that make (or update) the environment, in order."""
    python = str(env_python(env_dir))
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    steps = []
    if not env_python(env_dir).is_file():
        steps.append(Step(f"Creating the environment in {env_dir}",
                          [*base_python, "-m", "venv", str(env_dir)]))
    steps.append(Step("Updating pip", [*pip, "--upgrade", "pip"]))
    torch = [*pip, "torch", "torchvision"]
    words = {"cuda": "CUDA", "cpu": "CPU"}.get(build, "standard")
    steps.append(Step(f"Installing PyTorch ({words} build; a few GB the first time)",
                      [*torch, "--index-url", TORCH_INDEX[build]] if build in TORCH_INDEX else torch,
                      torch if build in TORCH_INDEX else None))
    steps.append(Step("Installing the lab's libraries",
                      [*pip, "-r", str(requirements or requirements_file())]))
    return steps


@dataclass(frozen=True)
class PythonFacts:
    python: str
    version: str = ""
    #: Import name → version ("" when present without a version).
    packages: dict[str, str] = field(default_factory=dict)
    cuda: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.version_tuple >= MINIMUM

    @property
    def version_tuple(self) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in self.version.split(".")[:2])
        except ValueError:
            return ()

    def missing(self) -> list[str]:
        """What this Python lacks, in words ("torch (training image models)")."""
        return [f"{name} ({purpose})" for name, purpose in NEEDED if name not in self.packages]


_PROBE = """
import importlib, json, sys
found, cuda = {}, False
for name in %r:
    try:
        module = importlib.import_module(name)
    except Exception:
        continue
    found[name] = str(getattr(module, "__version__", ""))
if "torch" in found:
    try:
        cuda = bool(sys.modules["torch"].cuda.is_available())
    except Exception:
        pass
print(json.dumps({"version": "%%d.%%d.%%d" %% sys.version_info[:3], "packages": found, "cuda": cuda}))
"""


def describe_python(python: str, timeout: float = 120) -> PythonFacts:
    """What ``python`` is and has: version, the lab's main libraries, whether CUDA works.
    Runs in a subprocess, so CUDA is never initialised in the app."""
    if not python:
        return PythonFacts("", error="no experiment Python is set")
    result = _run([python, "-c", _PROBE % (tuple(name for name, _ in NEEDED),)], timeout)
    if result is None:
        return PythonFacts(python, error=f"{python} could not be started")
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines() or ["it exited with an error"]
        return PythonFacts(python, error=lines[-1])
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return PythonFacts(python, error="it answered with something unexpected")
    facts = PythonFacts(python, data["version"], dict(data["packages"]), bool(data["cuda"]))
    if not facts.ok:
        return PythonFacts(python, facts.version, facts.packages, facts.cuda,
                           error=f"Python {facts.version} is too old; the lab needs "
                                 f"{'.'.join(map(str, MINIMUM))} or newer")
    return facts
