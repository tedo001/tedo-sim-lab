"""The reproducibility snapshot: everything outside ``experiment.yaml`` that can change a
result. Written to ``snapshot.json`` (and ``code.diff`` when the code has uncommitted
changes) in the run folder before training starts.

``compare`` lists what differs between two snapshots, so a reproduction can say up front
why its numbers might not match.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from .spec import ExperimentSpec, spec_hash

__all__ = ["PACKAGES", "compare", "dataset_fingerprint", "environment", "git_state", "take_snapshot"]

#: Packages whose version can change a result.
PACKAGES = ("torch", "torchvision", "numpy", "scikit-learn", "xgboost", "pandas", "Pillow", "mlflow")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def git_state(code_root: Path) -> dict[str, Any]:
    """Commit, branch and uncommitted changes of the code, or why they are unknown."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=code_root, capture_output=True, text=True, timeout=20,
                              check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        diff = git("diff", "HEAD", "--", ".")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "note": "not a git checkout (an installed build, or git is missing)"}
    return {"commit": commit, "branch": branch, "dirty": bool(diff.strip()), "diff": diff}


def environment(code_root: Path) -> dict[str, Any]:
    """This process's Python, platform, package versions and code state."""
    git = git_state(code_root)
    git.pop("diff", None)
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: _version(name) for name in PACKAGES}, "git": git}


def dataset_fingerprint(folder: Path) -> str | None:
    """A cheap fingerprint of a dataset folder: SHA-256 over every file's relative path and size."""
    if not folder.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".part")):
        digest.update(f"{path.relative_to(folder).as_posix()}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def take_snapshot(run_dir: Path, spec: ExperimentSpec, *, code_root: Path, datasets_root: Path,
                  device: str, runner: str) -> dict[str, Any]:
    git = git_state(code_root)
    diff = git.pop("diff", "")
    if diff.strip():
        (run_dir / "code.diff").write_text(diff, encoding="utf-8")
    snapshot = {"spec_hash": spec_hash(spec), "seed": spec.seed, "runner": runner, "device": device,
                "deterministic": spec.runtime.deterministic,
                "dataset": {"id": spec.data.dataset,
                            "fingerprint": dataset_fingerprint(datasets_root / spec.data.dataset)},
                **environment(code_root), "git": git}
    (run_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def compare(original: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Differences that could change a reproduced result, in words."""
    notes = []
    for key, title in (("python", "Python"), ("platform", "Platform")):
        if original.get(key) != current.get(key):
            notes.append(f"{title}: {original.get(key)} then, {current.get(key)} now")
    before, after = original.get("packages", {}), current.get("packages", {})
    for name in sorted(set(before) | set(after)):
        if before.get(name) != after.get(name):
            notes.append(f"{name}: {before.get(name) or 'not installed'} then, "
                         f"{after.get(name) or 'not installed'} now")
    old_git, new_git = original.get("git", {}), current.get("git", {})
    if old_git.get("commit") != new_git.get("commit"):
        then, now = str(old_git.get("commit"))[:10], str(new_git.get("commit"))[:10]
        notes.append(f"code: commit {then} then, {now} now")
    if old_git.get("dirty"):
        notes.append("the original run had uncommitted code changes (see its code.diff)")
    if "device" in original and "device" in current and original["device"] != current["device"]:
        notes.append(f"device: {original['device']} then, {current['device']} now")
    if not original.get("deterministic"):
        notes.append("the original run was not deterministic: expect small differences")
    return notes
