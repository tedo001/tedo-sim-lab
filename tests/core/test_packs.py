"""Environment packs: permissive licences only, a pack layered over the experiment Python sees its
packages without reinstalling them, and jobs run under a pack only once it is installed."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml
from packaging.requirements import Requirement

from core.common import packs
from core.common.licensing import PERMISSIVE_SPDX
from core.common.paths import CODE_ROOT

TABLE = yaml.safe_load((CODE_ROOT / "configs" / "dependency_licences.yaml").read_text())["packages"]


def test_every_pack_library_is_permissive_and_listed():
    known = {name.lower(): entry["spdx"] for name, entry in TABLE.items()}
    for pack in packs.load_packs().values():
        for requirement in pack.requirements:
            name = Requirement(requirement).name.lower()
            assert name in known, f"{name} ({pack.name}) is not in dependency_licences.yaml"
            assert known[name] in PERMISSIVE_SPDX, f"{name} is {known[name]}"
            assert pack.licences[Requirement(requirement).name] == known[name]


def test_a_layered_pack_and_who_runs_under_it(tmp_path, monkeypatch):
    monkeypatch.setenv(packs.PACKS_ENV, str(tmp_path / "packs"))
    pack = packs.Pack("demo", "Demo", "", ("PyYAML",), {"PyYAML": "MIT"}, ("sklearn",), True)
    assert packs.python_for(sys.executable, runner="sklearn", packs={"demo": pack}) == sys.executable
    for step in packs.install_steps(pack, sys.executable):
        done = subprocess.run(step.command, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=600)
        if done.returncode != 0 and "pip" in step.command:
            pytest.skip(f"pip could not run here: {done.stderr[-200:]}")
        assert done.returncode == 0, done.stderr
    python = packs.pack_python("demo")
    assert python is not None
    # The pack sees the experiment Python's packages (here numpy) without having them itself.
    seen = subprocess.run([str(python), "-c", "import numpy, sys; print(numpy.__file__); print(sys.prefix)"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert seen.returncode == 0, seen.stderr
    numpy_file, prefix = seen.stdout.splitlines()
    assert not numpy_file.startswith(prefix)
    assert packs.python_for(sys.executable, runner="sklearn", packs={"demo": pack}) == str(python)
    assert packs.python_for(sys.executable, inference=True, packs={"demo": pack}) == str(python)
    assert packs.python_for(sys.executable, runner="torch_classification", packs={"demo": pack}) == \
        sys.executable
    assert packs.describe(pack)["installed"] and packs.remove("demo")
    assert packs.pack_python("demo") is None
