"""The Windows installer's inputs stay consistent: what the window bundles, what it leaves to
the experiment Python, the licence notices, the permissive build tools, and where the installed
app finds its code (never on top of the app's compiled modules)."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from core.common.paths import CODE_ROOT

PACKAGING = CODE_ROOT / "packaging"
TRAINING_ONLY = {"torch", "torchvision", "xgboost", "opencv-python-headless", "matplotlib", "transformers"}


def _names(lines) -> set[str]:
    return {re.split(r"[<>=!~ \[]", line)[0].lower() for line in lines if line and not line.startswith("#")}


def test_window_requirements_follow_pyproject() -> None:
    project = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    window = _names((PACKAGING / "requirements-window.txt").read_text(encoding="utf-8").splitlines())
    assert window == _names(project["dependencies"]) - TRAINING_ONLY


def test_the_build_leaves_training_to_the_experiment_python() -> None:
    script = (PACKAGING / "freeze.py").read_text(encoding="utf-8")
    excludes = re.search(r"EXCLUDES = \[(.*?)\]", script, re.S).group(1)
    for name in ("torch", "torchvision", "xgboost", "cv2", "shap", "transformers"):
        assert f'"{name}"' in excludes
    assert 'f"lab-code/{folder}"' in script and '"configs"' in script and 'base="gui"' in script
    assert '"zip_exclude_packages": ["*"]' in script  # modules stay files: they read beside themselves
    assert '"all_users": False' in script  # installs without administrator rights


def test_no_copyleft_build_tool_is_left() -> None:
    for path in PACKAGING.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".bat", ".txt", ".md", ".iss", ".spec"} \
                and path.parent.name != "licences":
            text = path.read_text(encoding="utf-8").lower()
            assert "pyinstaller" not in text or path.name == "INSTALLER.md", path
    assert not list(PACKAGING.glob("*.spec")) and not list(PACKAGING.glob("*.iss"))


def test_the_notices_ship_with_the_app() -> None:
    licences = PACKAGING / "licences"
    # LGPL-3.0 must travel with the GPL-3.0 text it incorporates (its section 4).
    assert (licences / "LGPL-3.0.txt").is_file() and (licences / "GPL-3.0.txt").is_file()
    notices = (licences / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    assert "LGPL-3.0" in notices and "replace" in notices and "freeze-core (MIT)" in notices
    assert "lib/PySide6/" in notices and "PyInstaller" not in notices
    assert (PACKAGING / "tedo-lab.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_the_installed_app_reads_its_code_from_lab_code(monkeypatch, tmp_path) -> None:
    from core.common.paths import AppPaths, code_root

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "TEDO-AI-Lab.exe"))
    assert code_root() == tmp_path.resolve() / "lab-code"  # beside the executable
    workspace = AppPaths.resolve(env={}, code_root=code_root()).workspace
    assert workspace == (Path.home() / "TEDO-AI-Lab").resolve()  # never the install folder
    monkeypatch.undo()
    assert code_root() == CODE_ROOT
