"""What runs on the remote machine (standard library only until the lab's code is on the path):

    python <code>/core/remote/remote_main.py <code> <workspace> <command> [args…]

``unpack <archive> <folder>`` extracts an archive the dispatcher sent; ``pack <folder> <archive>``
archives a run folder to send back; ``exists <path>`` exits 0 when it is there; ``prepare
<experiment.yaml>`` makes the dataset available (downloads only what anyone may fetch); ``worker
<run folder>`` runs the experiment worker with the code and the remote workspace set up."""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path


def _safe_extract(archive: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (folder / member.name).resolve()
            if not str(target).startswith(str(folder.resolve())) or member.issym() or member.islnk():
                raise SystemExit(f"refusing {member.name!r}: it points outside {folder}")
        if hasattr(tarfile, "data_filter"):  # Python 3.12+ (and 3.11.4+): refuse anything unusual
            bundle.extractall(folder, filter="data")
        else:
            bundle.extractall(folder)


def main(argv: list[str]) -> int:
    code, workspace, command, *rest = argv
    code_path, workspace_path = Path(code).expanduser(), Path(workspace).expanduser()
    if command == "exists":
        return 0 if Path(rest[0]).expanduser().exists() else 3
    if command == "unpack":
        _safe_extract(Path(rest[0]).expanduser(), Path(rest[1]).expanduser())
        return 0
    if command == "pack":
        folder = Path(rest[0]).expanduser()
        with tarfile.open(Path(rest[1]).expanduser(), "w:gz") as bundle:
            bundle.add(folder, arcname=".")
        return 0
    sys.path.insert(0, str(code_path))
    os.environ["TEDO_LAB_WORKSPACE"] = str(workspace_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(code_path), os.environ.get("PYTHONPATH")]))
    workspace_path.mkdir(parents=True, exist_ok=True)
    os.chdir(code_path)
    if command == "prepare":
        from core.common.paths import AppPaths
        from core.experiment_engine.reproduce import _prepare
        from core.experiment_engine.spec import load_spec

        paths = AppPaths.resolve(workspace_path).ensure()
        problem = _prepare(load_spec(Path(rest[0]).expanduser()), paths, True)
        if problem:
            print(problem, file=sys.stderr)
            return 1
        return 0
    if command == "worker":
        from core.experiment_engine.worker import main as worker

        return worker([str(Path(rest[0]).expanduser()), "--cancel-on-eof", *rest[1:]])
    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
