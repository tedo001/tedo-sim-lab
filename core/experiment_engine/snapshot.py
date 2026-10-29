"""The reproducibility snapshot: everything outside ``experiment.yaml`` that can change a
result. Written to ``snapshot.json`` (and ``code.diff`` when the code has uncommitted
changes) in the run folder before training starts, with ``environment.lock.txt``: every installed
package at its exact version (``pip install -r`` rebuilds the environment).

The snapshot records Python, the platform and OS release, the CPU, the CUDA toolkit and cuDNN
PyTorch was built with, the GPUs and their driver (NVML), the key package versions and the code's
git state. ``compare`` lists what differs between two snapshots, so a reproduction can say up front
why its numbers might not match. ``python -m core.experiment_engine.snapshot`` prints the current
environment as JSON (the app asks the experiment Python this way, never itself).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from .spec import ExperimentSpec, spec_hash

__all__ = ["LOCK_FILE", "PACKAGES", "compare", "dataset_fingerprint", "environment", "environment_of",
           "git_state", "hardware", "package_lock", "take_snapshot"]

#: Packages whose version can change a result.
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "xgboost", "pandas", "Pillow", "mlflow",
            "transformers")
LOCK_FILE = "environment.lock.txt"


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def git_state(code_root: Path) -> dict[str, Any]:
    """Commit, branch and uncommitted changes of the code, or why they are unknown."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=code_root, capture_output=True, stdin=subprocess.DEVNULL,
                              text=True, timeout=20,
                              check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        diff = git("diff", "HEAD", "--", ".")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "note": "not a git checkout (an installed build, or git is missing)"}
    try:  # where the code can be cloned from; any credentials in the URL are dropped
        remote = git("config", "--get", "remote.origin.url").strip()
    except (OSError, subprocess.SubprocessError):
        remote = ""
    remote = re.sub(r"(?<=://)[^/@]*@", "", remote)
    return {"commit": commit, "branch": branch, "dirty": bool(diff.strip()), "remote": remote or None,
            "diff": diff}


def package_lock() -> list[str]:
    """Every installed distribution as ``name==version``, sorted (a pip requirements file)."""
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") if dist.metadata else None
        if name and dist.version:
            found.setdefault(name.lower(), f"{name}=={dist.version}")
    return [found[key] for key in sorted(found)]


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _torch_build(probe: bool) -> dict[str, Any]:
    """The CUDA toolkit (and, when ``probe``, cuDNN) PyTorch was built with; read without CUDA."""
    if _version("torch") is None:
        return {}
    if not probe:
        try:  # torch/version.py names the build without importing torch
            text = metadata.distribution("torch").locate_file("torch/version.py").read_text(encoding="utf-8")
        except (OSError, metadata.PackageNotFoundError):
            return {}
        line = next((line for line in text.splitlines() if line.startswith("cuda")), "")
        value = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else ""
        return {"cuda_toolkit": None if value in ("", "None") else value}
    try:
        import torch

        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        return {"cuda_toolkit": torch.version.cuda, "cudnn": cudnn, "torch_threads": torch.get_num_threads()}
    except Exception as exc:  # a broken install must not stop a run
        return {"cuda_toolkit": None, "note": f"torch could not be read: {type(exc).__name__}"}


def hardware() -> dict[str, Any]:
    """CPU model and cores; NVIDIA GPUs with memory and the driver, from NVML (no CUDA context)."""
    info: dict[str, Any] = {"cpu": _cpu(), "cpu_cores": os.cpu_count(), "gpus": [], "driver": None}
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            info["driver"] = driver.decode() if isinstance(driver, bytes) else driver
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                name = name.decode() if isinstance(name, bytes) else name
                info["gpus"].append({"index": index, "name": name, "memory_gb": round(memory / 1e9, 2)})
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # no NVIDIA driver, or NVML missing: a CPU machine
        pass
    return info


def environment(code_root: Path, *, probe_torch: bool = False) -> dict[str, Any]:
    """This process's Python, platform, hardware, package versions, PyTorch build and code state."""
    git = git_state(code_root)
    git.pop("diff", None)
    lock = package_lock()
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
            "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
            "hardware": hardware(), "torch_build": _torch_build(probe_torch),
            "packages": {name: _version(name) for name in PACKAGES},
            "lock": {"count": len(lock), "sha256": hashlib.sha256("\n".join(lock).encode()).hexdigest()},
            "git": git}


def environment_of(python: str, code_root: Path, timeout: float = 90) -> dict[str, Any]:
    """The environment of another interpreter (the experiment Python), asked in a subprocess; this
    process's own when ``python`` is empty or cannot answer."""
    if python:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(code_root),
                                                                         os.environ.get("PYTHONPATH", "")]))}
        try:
            result = subprocess.run([python, "-m", "core.experiment_engine.snapshot", str(code_root)],
                                    capture_output=True, text=True, timeout=timeout, env=env,
                                    stdin=subprocess.DEVNULL, cwd=str(code_root))
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return environment(code_root)


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
                **environment(code_root, probe_torch=True), "git": git}
    (run_dir / LOCK_FILE).write_text("\n".join(package_lock()) + "\n", encoding="utf-8")
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
    old_build, new_build = original.get("torch_build", {}), current.get("torch_build", {})
    if old_build.get("cuda_toolkit") != new_build.get("cuda_toolkit") and (old_build or new_build):
        notes.append(f"CUDA toolkit (PyTorch build): {old_build.get('cuda_toolkit') or 'none'} then, "
                     f"{new_build.get('cuda_toolkit') or 'none'} now")
    old_gpus = [g.get("name") for g in original.get("hardware", {}).get("gpus", [])]
    new_gpus = [g.get("name") for g in current.get("hardware", {}).get("gpus", [])]
    if "hardware" in original and "hardware" in current and old_gpus != new_gpus:
        notes.append(f"GPUs: {', '.join(old_gpus) or 'none'} then, {', '.join(new_gpus) or 'none'} now")
    old_lock, new_lock = original.get("lock", {}), current.get("lock", {})
    if old_lock.get("sha256") and new_lock.get("sha256") and old_lock["sha256"] != new_lock["sha256"]:
        notes.append(f"installed packages differ ({old_lock.get('count')} then, {new_lock.get('count')} now; "
                     f"compare {LOCK_FILE})")
    if "device" in original and "device" in current and original["device"] != current["device"]:
        notes.append(f"device: {original['device']} then, {current['device']} now")
    if not original.get("deterministic"):
        notes.append("the original run was not deterministic: expect small differences")
    return notes


if __name__ == "__main__":  # the experiment Python's environment, for comparisons made in the app
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    print(json.dumps(environment(root)))
