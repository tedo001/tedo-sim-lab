"""Owner policy: every dependency is permissively licensed (MIT, Apache-2.0, BSD ...).

Adding a package to pyproject.toml without a licence entry, or with a copyleft
licence, fails here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

from core.common.licensing import PERMISSIVE_SPDX

ROOT = Path(__file__).resolve().parents[1]
TABLE = yaml.safe_load((ROOT / "configs" / "dependency_licences.yaml").read_text())["packages"]


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared() -> dict[str, str]:
    """Every requirement in pyproject.toml, normalised name → group."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    found = {normalise(Requirement(req).name): "core" for req in project["dependencies"]}
    for group, requirements in project["optional-dependencies"].items():
        for req in requirements:
            found.setdefault(normalise(Requirement(req).name), group)
    return found


LICENCES = {normalise(name): entry for name, entry in TABLE.items()}


def permissive(spdx: str) -> bool:
    return any(option.strip() in PERMISSIVE_SPDX for option in spdx.split(" OR "))


@pytest.mark.parametrize("name", sorted(declared()))
def test_every_dependency_has_an_allowed_licence(name: str) -> None:
    assert name in LICENCES, f"{name}: add it to configs/dependency_licences.yaml with its licence"
    entry = LICENCES[name]
    if not permissive(entry["spdx"]):
        assert entry.get("exception"), f"{name} is {entry['spdx']}: not permissive, and no exception"


def test_no_copyleft_except_the_documented_qt_binding() -> None:
    exceptions = {name for name, entry in LICENCES.items() if not permissive(entry["spdx"])}
    assert exceptions == {"pyside6-essentials"}
    assert not any("AGPL" in entry["spdx"] for entry in LICENCES.values())


def test_banned_packages_are_absent() -> None:
    for banned in ("ultralytics", "pyqt6", "pyqt5"):
        assert banned not in declared(), f"{banned} is not allowed (licence policy)"


def test_table_has_no_stale_entries() -> None:
    assert set(LICENCES) <= set(declared()), set(LICENCES) - set(declared())
