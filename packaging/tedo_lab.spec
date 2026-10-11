# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of TEDO AI Research Lab: the window only. Build from the repository root
(PyInstaller builds for the system it runs on, so the Windows .exe is built on Windows):

    pyinstaller packaging/tedo_lab.spec --noconfirm --clean

Output: ``dist/TEDO-AI-Lab/TEDO-AI-Lab(.exe)`` and the folder it runs from (one folder, not one
file: Qt (LGPL-3.0) stays a set of separate, replaceable libraries). ``packaging/installer.iss``
turns that folder into a Windows setup file; ``packaging/build_installer.bat`` does both.

What is in the bundle: the interface and what it reads in-process (PySide6, numpy, pandas,
scikit-learn for its bundled tables, MLflow for the MLflow page, nltk, pycocotools, keyring).
What is not: PyTorch, torchvision, XGBoost, SHAP, OpenCV. Training, evaluation and benchmarks
run in the *experiment Python*, which the app makes on request (Settings → Experiment Python).
That Python imports the lab's code from ``lab-code/``: plain sources, cards and documents,
nothing compiled, so any Python 3.11+ can use it.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(os.getcwd()).resolve()
APP = "TEDO-AI-Lab"
ICON = ROOT / "packaging" / "tedo-lab.ico"

# The lab's own packages: pages, plugins and runners are reached through strings
# (NAV, plugin.yaml, runners.yaml), so they are listed rather than discovered.
hidden = []
for package in ("app", "core", "labs", "plugins"):
    hidden += [package] + collect_submodules(package, filter=lambda name: ".tests" not in name)
hidden += ["sqlalchemy.dialects.sqlite", "keyring.backends", "keyring.backends.Windows",
           "keyring.backends.macOS", "keyring.backends.SecretService", "keyring.backends.fail"]

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def tree(source: Path, target: str) -> list[tuple[str, str]]:
    """Every file under ``source`` (caches left out) as PyInstaller datas under ``target``."""
    found = []
    for path in source.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(source).parts) and path.suffix != ".pyc":
            found.append((str(path), str(Path(target) / path.relative_to(source).parent)))
    return found


datas = [
    # The interface's own resources, next to the modules that find them by __file__.
    *tree(ROOT / "app" / "resources", "app/resources"),
    (str(ROOT / "app" / "ui" / "theme" / "style.qss"), "app/ui/theme"),
    # lab-code/: the code root of the installed app (see core/common/paths.py).
    *tree(ROOT / "core", "lab-code/core"),
    *tree(ROOT / "labs", "lab-code/labs"),
    *tree(ROOT / "plugins", "lab-code/plugins"),
    *tree(ROOT / "configs", "lab-code/configs"),
    *tree(ROOT / "docs", "lab-code/docs"),
    *[(str(ROOT / name), "lab-code") for name in ("README.md", "ROADMAP.md", "CLAUDE.md")],
    # Files the lab's modules read from beside themselves in-process (database migrations,
    # the ImageNet class list, licence notices), at the same place as the frozen modules.
    *[item for package in ("core", "labs", "plugins")
      for item in tree(ROOT / package, package) if not item[0].endswith(".py")],
    *tree(ROOT / "packaging" / "licences", "licences"),
]
binaries = []
for package in ("sklearn", "mlflow", "nltk", "pycocotools"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception as exc:  # not installed on this build machine: the page says so at run time
        print(f"[tedo_lab.spec] {package} not collected: {exc}")
        continue
    datas += package_datas
    binaries += package_binaries
    hidden += package_hidden
for distribution in ("keyring", "mlflow", "scikit-learn", "PySide6-Essentials", "numpy", "pandas"):
    try:
        datas += copy_metadata(distribution)  # version checks read distribution metadata
    except Exception as exc:
        print(f"[tedo_lab.spec] no metadata for {distribution}: {exc}")

# Training libraries live in the experiment Python; the rest are optional extras of MLflow and
# pandas that the window never uses (several hundred MB together).
excludes = ["torch", "torchvision", "torchaudio", "xgboost", "shap", "cv2", "tensorflow", "jax",
            "numba", "llvmlite", "onnxruntime", "pyarrow", "matplotlib", "IPython", "jupyterlab",
            "notebook", "tkinter", "PyQt5", "PyQt6", "pytest", "ruff"]

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hidden)),
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    debug=False,
    strip=False,
    upx=False,
    console=False,  # a desktop application: no terminal window
    icon=str(ICON) if ICON.is_file() and sys.platform == "win32" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP)
