"""The Windows installer's inputs stay consistent: what the window bundles, what it leaves to
the experiment Python, the files the setup script names, the licence notices, and where the
installed app finds its code (never on top of the bundle's compiled libraries)."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from core.common.paths import CODE_ROOT

PACKAGING = CODE_ROOT / "packaging"
TRAINING_ONLY = {"torch", "torchvision", "xgboost", "opencv-python-headless", "matplotlib"}


def _names(lines) -> set[str]:
    return {re.split(r"[<>=!~ \[]", line)[0].lower() for line in lines if line and not line.startswith("#")}


def test_window_requirements_follow_pyproject() -> None:
    project = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    window = _names((PACKAGING / "requirements-window.txt").read_text(encoding="utf-8").splitlines())
    assert window == _names(project["dependencies"]) - TRAINING_ONLY


def test_the_spec_leaves_training_to_the_experiment_python() -> None:
    spec = (PACKAGING / "tedo_lab.spec").read_text(encoding="utf-8")
    excludes = re.search(r"excludes = \[(.*?)\]", spec, re.S).group(1)
    for name in ("torch", "torchvision", "xgboost", "cv2", "shap"):
        assert f'"{name}"' in excludes
    assert "lab-code/core" in spec and "lab-code/configs" in spec and "console=False" in spec
    assert "COLLECT(" in spec  # one folder: Qt stays a set of replaceable libraries (LGPL)


def test_the_setup_script_names_files_that_exist() -> None:
    script = (PACKAGING / "installer.iss").read_text(encoding="utf-8")
    for name in re.findall(r"(?:InfoBeforeFile|InfoAfterFile|SetupIconFile)=(\S+)", script):
        assert (PACKAGING / name).is_file(), name
    assert 'Source: "licences\\*"' in script and (PACKAGING / "licences" / "LGPL-3.0.txt").is_file()
    assert "{autopf}" in script  # Program Files, never the workspace
    notices = (PACKAGING / "licences" / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    assert "LGPL-3.0" in notices and "replace" in notices and "PyInstaller" in notices
    assert (PACKAGING / "tedo-lab.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_the_installed_app_reads_its_code_from_lab_code(monkeypatch, tmp_path) -> None:
    from core.common.paths import AppPaths, code_root

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert code_root() == tmp_path / "lab-code"
    workspace = AppPaths.resolve(env={}, code_root=code_root()).workspace
    assert workspace == (Path.home() / "TEDO-AI-Lab").resolve()  # never the install folder
    monkeypatch.undo()
    assert code_root() == CODE_ROOT
