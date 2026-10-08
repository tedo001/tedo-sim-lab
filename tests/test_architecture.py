"""The ground rules, checked mechanically.

* ``core`` imports no Qt and nothing from ``app``, ``labs`` or ``plugins``;
* ``labs`` and ``plugins`` import neither Qt nor ``app``;
* no Python file is over 400 lines.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_LINES = 400
QT = ("PyQt6", "PyQt5", "PySide6", "PySide2", "pyqtgraph", "pytestqt")

FORBIDDEN = {
    "core": QT + ("app", "labs", "plugins"),
    "labs": QT + ("app",),
    "plugins": QT + ("app",),
}


def python_files(*packages: str) -> list[Path]:
    return sorted(file for package in packages for file in (ROOT / package).rglob("*.py"))


def imported_modules(file: Path) -> set[str]:
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_layer_imports(package: str) -> None:
    banned = FORBIDDEN[package]
    offences = [
        f"{file.relative_to(ROOT)} imports {name}"
        for file in python_files(package)
        for name in sorted(imported_modules(file))
        if name.split(".")[0] in banned
    ]
    assert not offences, "layering rule broken:\n" + "\n".join(offences)


def test_no_file_over_400_lines() -> None:
    long_files = []
    for file in python_files("app", "core", "labs", "plugins", "tests"):
        count = len(file.read_text(encoding="utf-8").splitlines())
        if count > MAX_LINES:
            long_files.append(f"{file.relative_to(ROOT)}: {count} lines")
    assert not long_files, "split these files:\n" + "\n".join(long_files)


def test_the_checker_sees_imports(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("import PyQt6.QtCore\nfrom app.main import main\nfrom . import sibling\n")
    assert imported_modules(sample) == {"PyQt6.QtCore", "app.main"}
