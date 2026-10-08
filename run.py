#!/usr/bin/env python3
"""Start TEDO AI Research Lab with one command:

    python run.py                      (Windows: double-click run.bat)

The first run makes a private Python environment in ``.venv/`` next to this file, installs
PyTorch (the CUDA build when an NVIDIA GPU is found, otherwise the CPU build) and the lab,
then opens the app. Later runs open it straight away; they reinstall only when
``pyproject.toml`` changed. Arguments go to the app, e.g. ``python run.py --workspace D:/lab``.

Options of its own:
    --setup-only   install (or update) and stop, without opening the app
    --reinstall    install again even if nothing changed
    --cpu          use the CPU build of PyTorch even when a GPU is present

Needs Python 3.11 or newer. Uses only the standard library, so it runs before anything is
installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / "tedo-installed.json"
MINIMUM = (3, 11)
#: PyTorch's own package indexes. CUDA 12.6 wheels run on any NVIDIA driver made for CUDA 12.
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu126",
               "cpu": "https://download.pytorch.org/whl/cpu"}
LINUX_QT = "sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3"
OWN_OPTIONS = ("--setup-only", "--reinstall", "--cpu")


def say(text: str) -> None:
    print(f"[tedo] {text}", flush=True)


def venv_python(venv: Path = VENV) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def has_nvidia_gpu() -> bool:
    """True when ``nvidia-smi`` exists and lists a GPU (no PyTorch needed to ask)."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return False
    try:
        result = subprocess.run([tool, "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def torch_build(force_cpu: bool, system: str | None = None, gpu: bool | None = None) -> str:
    """'cuda', 'cpu', or 'default' (macOS: the standard wheels already use Apple's GPU)."""
    system = system or platform.system()
    if system == "Darwin":
        return "default"
    if force_cpu:
        return "cpu"
    return "cuda" if (has_nvidia_gpu() if gpu is None else gpu) else "cpu"


def fingerprint(build: str) -> dict[str, str]:
    digest = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
    return {"pyproject": digest, "torch": build, "python": platform.python_version()}


def run(command: list[str], **kwargs) -> int:  # noqa: ANN003 (passed to subprocess)
    return subprocess.call(command, cwd=ROOT, **kwargs)


def make_venv() -> None:
    say(f"Creating a Python environment in {VENV} …")
    if run([sys.executable, "-m", "venv", str(VENV)]) != 0:
        linux = sys.platform.startswith("linux")
        hint = " On Debian/Ubuntu: sudo apt-get install python3-venv" if linux else ""
        sys.exit(f"[tedo] Could not create the environment.{hint}")


def install(build: str) -> None:
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    say(f"Installing PyTorch ({'CUDA' if build == 'cuda' else 'CPU' if build == 'cpu' else 'standard'} "
        "build); the first time this downloads a few GB …")
    torch = [python, "-m", "pip", "install", "torch", "torchvision"]
    if build in TORCH_INDEX and run([*torch, "--index-url", TORCH_INDEX[build]]) == 0:
        pass
    elif run(torch) != 0:
        sys.exit("[tedo] PyTorch could not be installed; check the internet connection and try again.")
    say("Installing the lab and its libraries …")
    if run([python, "-m", "pip", "install", "-e", "."]) != 0:
        sys.exit("[tedo] Installing the lab failed; the messages above say why.")


def main(argv: list[str]) -> int:
    if sys.version_info < MINIMUM:
        say(f"Python {'.'.join(map(str, MINIMUM))} or newer is needed; this is {platform.python_version()}. "
            "Get it from https://www.python.org/downloads/ and run this file with it.")
        return 1
    options = {name for name in OWN_OPTIONS if name in argv}
    app_args = [arg for arg in argv if arg not in OWN_OPTIONS]
    if not venv_python().is_file():
        make_venv()
    build = torch_build("--cpu" in options)
    wanted = fingerprint(build)
    installed = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.is_file() else None
    if installed != wanted or "--reinstall" in options:
        install(build)
        MARKER.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
        say("Ready.")
    if "--setup-only" in options:
        return 0
    say("Starting TEDO AI Research Lab …")
    code = run([str(venv_python()), "-m", "app.main", *app_args])
    if code != 0 and sys.platform.startswith("linux"):
        say(f"If Qt could not start, install its system libraries: {LINUX_QT}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
