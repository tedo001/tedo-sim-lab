"""Build TEDO AI Research Lab with cx_Freeze (PSF-2.0; its launcher, freeze-core, is MIT), run
from the repository root:

    python packaging/freeze.py build_exe     # dist/TEDO-AI-Lab/: the app folder
    python packaging/freeze.py bdist_msi     # Windows: dist/installer/TEDO-AI-Lab-<version>-win64.msi

cx_Freeze builds for the system it runs on, so the Windows app and its .msi are built on
Windows. The .msi is written with python-msilib (PSF-2.0) and installed by Windows Installer,
which is part of Windows: no other installer tool is involved.

What is in the app: the interface and what it reads in-process (PySide6, numpy, pandas,
scikit-learn for its bundled tables, MLflow for the MLflow page, nltk, pycocotools, keyring),
in one folder so Qt (LGPL-3.0) stays a set of separate, replaceable libraries. What is not:
PyTorch, torchvision, XGBoost, SHAP, OpenCV. Training, evaluation and benchmarks run in the
*experiment Python*, which the app makes on request (Settings → Experiment Python). That Python
imports the lab's code from ``lab-code/``: plain sources, cards and documents, nothing compiled,
so any Python 3.11+ can use it.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from cx_Freeze import Executable, setup
from cx_Freeze.command.build_exe import build_exe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402

APP = "TEDO-AI-Lab"
PRODUCT = "TEDO AI Research Lab"
ICON = ROOT / "packaging" / "tedo-lab.ico"
#: Never change it: Windows Installer recognises a newer version of the same product by it and
#: replaces the old one.
UPGRADE_CODE = "{6B0E3B52-6A2D-4C1E-9E7B-2F1D7C4A9B31}"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
WINDOWS = sys.platform == "win32"

# The lab's own packages, whole: pages, plugins and runners are reached through strings (NAV,
# plugin.yaml, runners.yaml), so nothing finds them by following imports. The other packages
# load parts of themselves by name too.
PACKAGES = ["app", "core", "labs", "plugins", "sklearn", "mlflow", "nltk", "pycocotools",
            "keyring.backends", "sqlalchemy.dialects.sqlite", "pyqtgraph", "psutil"]
if sys.platform == "win32":
    PACKAGES.append("winpty")  # the terminal's ConPTY console

# Training libraries live in the experiment Python; the rest are optional extras of MLflow and
# pandas that the window never uses (several hundred MB together).
EXCLUDES = ["torch", "torchvision", "torchaudio", "xgboost", "shap", "cv2", "tensorflow", "jax",
            "numba", "llvmlite", "onnxruntime", "pyarrow", "matplotlib", "IPython", "jupyterlab",
            "notebook", "tkinter", "PyQt5", "PyQt6", "pytest", "_pytest", "ruff", "cx_Freeze",
            "sympy", "debugpy", "cuda", "claude_agent_sdk", "transformers", "tokenizers",
            "safetensors"]


def tree(source: Path, target: str, *, data_only: bool = False) -> list[tuple[str, str]]:
    """Every file under ``source`` (caches left out; with ``data_only`` no .py files) as
    cx_Freeze ``include_files`` entries below ``target`` in the app folder."""
    found = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (not path.is_file() or SKIP_DIRS & set(relative.parts) or path.suffix == ".pyc"
                or (data_only and path.suffix == ".py")):
            continue
        found.append((str(path), str(Path(target) / relative)))
    return found


def metadata_folders() -> list[tuple[str, str]]:
    """The ``*.dist-info`` folders of the window's requirements and everything they depend on:
    version checks, keyring's backends and MLflow's plugins read package metadata."""
    from importlib.metadata import PackageNotFoundError, distribution

    from packaging.requirements import Requirement

    lines = (ROOT / "packaging" / "requirements-window.txt").read_text(encoding="utf-8").splitlines()
    waiting = [Requirement(line).name for line in lines if line.strip() and not line.startswith("#")]
    seen: dict[str, Path] = {}
    while waiting:
        name = waiting.pop().lower().replace("_", "-")
        if name in seen or name in {item.lower() for item in EXCLUDES}:
            continue  # excluded packages must not look installed
        try:
            found = distribution(name)
        except PackageNotFoundError:
            continue
        folder = getattr(found, "_path", None)
        if folder is None or Path(folder).suffix != ".dist-info":
            continue
        seen[name] = Path(folder)
        for text in found.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                waiting.append(requirement.name)
    return [item for folder in seen.values() for item in tree(folder, f"lib/{folder.name}")]


def include_files() -> list[tuple[str, str]]:
    files = [
        # Files the lab's modules read from beside themselves in-process (the interface's fonts,
        # icons and style sheet, database migrations, the ImageNet class list, licence notices),
        # next to the compiled modules in lib/.
        *[item for package in ("app", "core", "labs", "plugins")
          for item in tree(ROOT / package, f"lib/{package}", data_only=True)],
        # lab-code/: the code root of the installed app (see core/common/paths.py).
        *[item for folder in ("core", "labs", "plugins", "configs", "docs")
          for item in tree(ROOT / folder, f"lab-code/{folder}")],
        *[(str(ROOT / name), f"lab-code/{name}") for name in ("README.md", "ROADMAP.md", "CLAUDE.md")],
        *tree(ROOT / "packaging" / "licences", "licences"),
        *metadata_folders(),
    ]
    return files


class BuildExe(build_exe):
    """cx_Freeze's build, then without the package metadata it adds to ``lib/library.zip`` on its
    own: it adds it for every distribution it looked at, the excluded ones too (PyTorch,
    XGBoost), and the app decides what is installed from metadata. What the app really holds is
    described by :func:`metadata_folders`, in ``lib/``."""

    def run(self) -> None:
        super().run()
        library = Path(self.build_exe) / "lib" / "library.zip"
        if not library.is_file():
            return
        with zipfile.ZipFile(library) as source:
            kept = [(item, source.read(item)) for item in source.infolist()
                    if ".dist-info" not in item.filename.split("/")[0]]
        with zipfile.ZipFile(library, "w", zipfile.ZIP_DEFLATED) as target:
            for item, data in kept:
                target.writestr(item, data)


executable = Executable(
    script=ROOT / "packaging" / "launch.py",
    base="gui" if WINDOWS else None,  # a desktop application: no terminal window on Windows
    target_name=APP,
    icon=ICON if WINDOWS else None,
    shortcut_name=PRODUCT,
    shortcut_dir="ProgramMenuFolder",
    copyright="Copyright (c) tedo001",
)

setup(
    name=APP,
    version=__version__,
    description=PRODUCT,
    author="tedo001",
    author_email="durgamani.d.e.c.e.50@gmail.com",
    url="https://github.com/tedo001/tedo-sim-lab",
    executables=[executable],
    cmdclass={"build_exe": BuildExe},
    options={
        "build_exe": {
            "build_exe": str(ROOT / "dist" / APP),
            "packages": PACKAGES,
            "excludes": EXCLUDES,
            "include_files": include_files(),
            # Modules stay plain files in lib/, never in a zip: several read files beside
            # themselves through __file__.
            "zip_include_packages": [],
            "zip_exclude_packages": ["*"],
            "include_msvcr": True,
            "silent_level": 1,
        },
        "bdist_msi": {
            "dist_dir": str(ROOT / "dist" / "installer"),
            "output_name": f"{APP}-{__version__}-win64.msi",
            "product_name": PRODUCT,
            "upgrade_code": UPGRADE_CODE,
            # For the person installing, without administrator rights (MSIINSTALLPERUSER).
            "all_users": False,
            "initial_target_dir": rf"[LocalAppDataFolder]Programs\{PRODUCT}",
            "install_icon": str(ICON),
            "launch_on_finish": True,
            "summary_data": {"author": "tedo001",
                             "comments": "Machine-learning and computer-vision research workbench"},
        },
    },
)
