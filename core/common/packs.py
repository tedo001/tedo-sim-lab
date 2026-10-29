"""Environment packs: heavy, optional libraries (Transformers, SHAP, ONNX) live in their own
virtual environments instead of the core experiment Python, and are installed on request.

A pack is a venv made from the experiment Python, with a ``.pth`` file that puts the experiment
Python's site-packages on its path *after* its own: everything in the core environment (PyTorch,
scikit-learn …) is available without being installed twice, and the pack's own packages take
precedence. Packs live per user beside the experiment Python (``…/tedo-ai-lab/packs/<name>``;
``TEDO_LAB_PACKS`` moves them). Which Python a job runs under: the pack named for its runner in
``configs/packs.yaml`` when that pack is installed, else the experiment Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .experiment_env import Step, default_env_dir, env_python
from .paths import CODE_ROOT

__all__ = ["LINK_FILE", "PACKS_ENV", "Pack", "install_steps", "load_packs", "pack_dir", "pack_python",
           "packs_root", "python_for", "remove", "site_packages", "describe"]

PACKS_ENV = "TEDO_LAB_PACKS"
LINK_FILE = "tedo_core_environment.pth"


@dataclass(frozen=True)
class Pack:
    name: str
    title: str
    description: str
    requirements: tuple[str, ...]
    licences: dict[str, str]
    runners: tuple[str, ...] = ()
    inference: bool = False


def load_packs(code_root: Path = CODE_ROOT) -> dict[str, Pack]:
    data = yaml.safe_load((code_root / "configs" / "packs.yaml").read_text(encoding="utf-8")) or {}
    return {name: Pack(name, entry["title"], entry["description"], tuple(entry["requirements"]),
                       dict(entry.get("licences", {})), tuple(entry.get("runners", ())),
                       bool(entry.get("inference")))
            for name, entry in (data.get("packs") or {}).items()}


def packs_root() -> Path:
    return Path(os.environ[PACKS_ENV]) if os.environ.get(PACKS_ENV) else default_env_dir().parent / "packs"


def pack_dir(name: str) -> Path:
    return packs_root() / name


def pack_python(name: str) -> Path | None:
    """The pack's interpreter once it is installed (made and linked), else ``None``."""
    python = env_python(pack_dir(name))
    return python if python.is_file() and (pack_dir(name) / "tedo-pack.json").is_file() else None


def site_packages(python: str, timeout: float = 60) -> list[str]:
    """Where ``python`` keeps its packages (asked in a subprocess)."""
    code = ("import json, os, site, sysconfig; "
            "paths = [sysconfig.get_paths()['purelib'], *site.getsitepackages()]; "
            "print(json.dumps([p for p in dict.fromkeys(paths) if os.path.isdir(p)]))")
    done = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        raise RuntimeError(f"{python} could not say where its packages are: {done.stderr.strip()[-200:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


_LINK = """
import json, sys, sysconfig
from pathlib import Path
own = sysconfig.get_paths()["purelib"]
Path(own, sys.argv[2]).write_text("\\n".join(json.loads(sys.argv[1])) + "\\n", encoding="utf-8")
"""
_DONE = """
import json, sys
from pathlib import Path
Path(sys.argv[1], "tedo-pack.json").write_text(json.dumps({"pack": sys.argv[2], "base": sys.argv[3]}),
                                               encoding="utf-8")
"""


def install_steps(pack: Pack, base_python: str) -> list[Step]:
    """Make the pack's venv from ``base_python``, link it to the base's packages, install the pack."""
    folder = pack_dir(pack.name)
    python = str(env_python(folder))
    base = site_packages(base_python)
    steps = []
    if not env_python(folder).is_file():
        steps.append(Step(f"Creating the {pack.name} environment in {folder}",
                          [base_python, "-m", "venv", str(folder)]))
    steps.append(Step("Linking it to the experiment Python's packages",
                      [python, "-c", _LINK, json.dumps(base), LINK_FILE]))
    steps.append(Step(f"Installing {', '.join(pack.requirements)}",
                      [python, "-m", "pip", "install", "--disable-pip-version-check", *pack.requirements]))
    steps.append(Step("Marking the pack as installed",
                      [python, "-c", _DONE, str(folder), pack.name, base_python]))
    return steps


def remove(name: str) -> bool:
    folder = pack_dir(name)
    if folder.is_dir():
        shutil.rmtree(folder)
        return True
    return False


def python_for(base_python: str, *, runner: str | None = None, inference: bool = False,
               packs: dict[str, Pack] | None = None) -> str:
    """The interpreter for a job: an installed pack that serves it, else ``base_python``."""
    for pack in (packs or load_packs()).values():
        wanted = (runner is not None and runner in pack.runners) or (inference and pack.inference)
        python = pack_python(pack.name) if wanted else None
        if python is not None:
            return str(python)
    return base_python


def describe(pack: Pack) -> dict[str, Any]:
    folder = pack_dir(pack.name)
    installed = pack_python(pack.name) is not None
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if installed else 0
    return {"name": pack.name, "installed": installed, "folder": str(folder), "size_mb": size / 1e6}
