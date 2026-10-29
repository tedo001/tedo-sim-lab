"""Run an experiment on a remote target, standing in for the local worker:

    python -m core.remote.dispatch <run folder> --target <name> --workspace <local workspace>
                                   [--prepare-data] [--cancel-on-eof]

1. sends the lab's code (``core``, ``labs``, ``plugins``, ``configs``; once per version, by content
   hash) and the run folder to ``<target folder>/code/<hash>`` and ``<target folder>/runs/<run id>``;
2. with ``--prepare-data``, makes the dataset available there (downloads only what anyone may fetch);
3. runs the experiment worker there and passes its JSON events through unchanged, so the app
   follows a remote run like a local one (its stderr goes to stderr); a ``cancel`` line on stdin is
   sent on, and closing stdin cancels the remote run;
4. brings the whole run folder back (checkpoints, metrics, snapshot, report) into the local run folder
   and exits with the remote worker's code (0 done, 1 failed, 2 cancelled).

The snapshot and report are written on the remote machine, so they record *its* environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

from core.common.masking import mask_text
from core.common.paths import CODE_ROOT
from core.common.stdio import detach_stdin
from core.experiment_engine.events import JsonLinesCallbacks

from .targets import Target, TargetError, load_targets
from .transport import Transport, transport_for

__all__ = ["CODE_PARTS", "code_digest", "dispatch", "main"]

CODE_PARTS = ("core", "labs", "plugins", "configs")
#: Written by the app as the run's events arrive; never sent, never overwritten.
LOCAL_ONLY = "run.log"
_SKIP = ("__pycache__", ".pytest_cache")


def _files(code_root: Path) -> Iterable[Path]:
    for part in CODE_PARTS:
        for path in sorted((code_root / part).rglob("*")):
            if path.is_file() and not any(skip in path.parts for skip in _SKIP) and path.suffix != ".pyc":
                yield path


def code_digest(code_root: Path = CODE_ROOT) -> str:
    """A hash of the code that would be sent: the same code is sent once."""
    digest = hashlib.sha256()
    for path in _files(code_root):
        digest.update(path.relative_to(code_root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def _pack(paths: Iterable[Path], root: Path, archive: Path) -> Path:
    with tarfile.open(archive, "w:gz") as bundle:
        for path in paths:
            bundle.add(path, arcname=path.relative_to(root).as_posix())
    return archive


def _localise(line: str, run_dir: Path) -> str:
    """The remote worker's event with its paths turned into this machine's run folder."""
    try:
        event = json.loads(line)
    except ValueError:
        return line
    marker = f"runs/{run_dir.name}"
    payload = event.get("payload") if isinstance(event, dict) else None
    if not isinstance(payload, dict):
        return line
    for key, value in payload.items():
        if isinstance(value, str) and marker in value.replace("\\", "/"):
            rest = value.replace("\\", "/").split(marker, 1)[1].lstrip("/")
            payload[key] = str(run_dir / rest) if rest else str(run_dir)
    return json.dumps(event, separators=(",", ":")) + "\n"


def _remote(transport: Transport, target: Target, code: str, *args: str) -> list[str]:
    return [target.python, f"{code}/core/remote/remote_main.py", code, transport.path("workspace"), *args]


def _check(done, what: str) -> None:  # noqa: ANN001 (CompletedProcess)
    if done.returncode != 0:
        raise RuntimeError(f"{what} failed on the target: {(done.stderr or done.stdout).strip()[-400:]}")


def dispatch(run_dir: Path, target: Target, *, prepare_data: bool = False, stdin=None,  # noqa: ANN001
             cancel_on_eof: bool = False, extra: tuple[str, ...] = (), log=print) -> int:  # noqa: ANN001
    """Run ``run_dir`` on ``target``; returns the remote worker's exit code."""
    run_dir = Path(run_dir).resolve()
    transport = transport_for(target)
    digest = code_digest()
    code = transport.path("code", digest)
    remote_run = transport.path("runs", run_dir.name)
    with tempfile.TemporaryDirectory(prefix="tedo-dispatch-") as scratch:
        scratch_dir = Path(scratch)
        if target.transport == "ssh":
            _check(transport.run(["mkdir", "-p", transport.path("code"), transport.path("runs"),
                                  transport.path("incoming")]), "Making the folders")
        else:
            Path(transport.path("incoming")).mkdir(parents=True, exist_ok=True)
        if transport.run(_remote(transport, target, code, "exists", f"{code}/core")).returncode != 0:
            log(f"Sending the lab's code ({digest}) to {target.name}")
            archive = _pack(_files(CODE_ROOT), CODE_ROOT, scratch_dir / "code.tar.gz")
            incoming = transport.path("incoming", f"code-{digest}.tar.gz")
            transport.put(archive, incoming)
            unpack = [target.python, "-c", _BOOT, incoming, code]  # remote_main is not there yet
            _check(transport.run(unpack), "Unpacking the code")
        log(f"Sending the run folder to {target.name}")
        sent = [p for p in run_dir.rglob("*") if p.is_file() and p.name != LOCAL_ONLY]
        archive = _pack(sent, run_dir, scratch_dir / "run.tar.gz")
        incoming = transport.path("incoming", f"run-{run_dir.name}.tar.gz")
        transport.put(archive, incoming)
        _check(transport.run(_remote(transport, target, code, "unpack", incoming, remote_run)),
               "Unpacking the run")
        if prepare_data:
            log("Making the dataset available on the target")
            prepare = _remote(transport, target, code, "prepare", f"{remote_run}/experiment.yaml")
            _check(transport.run(prepare, timeout=7200), "Preparing the data")
        log(f"Running on {target.name} ({target.describe()})")
        # Worker options that name a file of this run (--resume, --evaluate) name it there instead.
        passed = [f"{remote_run}/{Path(arg).resolve().relative_to(run_dir).as_posix()}"
                  if Path(arg).is_absolute() and Path(arg).resolve().is_relative_to(run_dir) else arg
                  for arg in extra]
        process = transport.popen(_remote(transport, target, code, "worker", remote_run, *passed))
        errors = threading.Thread(target=lambda: [sys.stderr.write(line) for line in process.stderr],
                                  daemon=True)
        errors.start()
        if stdin is not None:
            threading.Thread(target=_forward, args=(stdin, process, cancel_on_eof), daemon=True).start()
        for line in process.stdout:  # the remote worker's events, with this machine's paths
            sys.stdout.write(_localise(line, run_dir))
            sys.stdout.flush()
        code_exit = process.wait()
        errors.join(timeout=5)
        log(f"Bringing the run folder back from {target.name}")
        packed = transport.path("incoming", f"result-{run_dir.name}.tar.gz")
        _check(transport.run(_remote(transport, target, code, "pack", remote_run, packed)),
               "Packing the run")
        local = scratch_dir / "result.tar.gz"
        transport.get(packed, local)
        _unpack_result(local, run_dir)
    return code_exit


def _unpack_result(archive: Path, run_dir: Path) -> None:
    with tarfile.open(archive) as bundle:
        members = []
        for member in bundle.getmembers():
            inside = (run_dir / member.name).resolve().is_relative_to(run_dir)
            if member.issym() or member.islnk() or not inside:
                raise RuntimeError(f"refusing {member.name!r} from the target")
            if Path(member.name).name != LOCAL_ONLY:  # the app's own log of this run stays as it is
                members.append(member)
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(run_dir, members=members, filter="data")
        else:
            bundle.extractall(run_dir, members=members)


def _forward(stream, process, close_on_eof: bool) -> None:  # noqa: ANN001
    """Pass cancel lines on; with ``close_on_eof``, the end of our stdin (the app went away) closes
    the remote one, and the remote worker cancels."""
    try:
        for line in stream:
            process.stdin.write(line)
            process.stdin.flush()
    except (OSError, ValueError):
        pass
    if close_on_eof:
        try:
            process.stdin.close()
        except OSError:
            pass


#: Unpacks the first code archive with the standard library only (nothing of the lab is there yet).
_BOOT = ("import sys, tarfile; from pathlib import Path; a, d = Path(sys.argv[1]).expanduser(), "
         "Path(sys.argv[2]).expanduser(); d.mkdir(parents=True, exist_ok=True); t = tarfile.open(a); "
         "bad = [m.name for m in t.getmembers() if m.issym() or m.islnk() or '..' in Path(m.name).parts]; "
         "sys.exit('refusing ' + ', '.join(bad)) if bad else "
         "(t.extractall(d, filter='data') if hasattr(tarfile, 'data_filter') else t.extractall(d))")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="core.remote.dispatch")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--cancel-on-eof", action="store_true")
    args, extra = parser.parse_known_args(argv)  # the rest goes to the remote worker
    stream = detach_stdin()  # (Windows: see core.common.stdio)
    events = JsonLinesCallbacks(sys.stdout, args.run_dir.name)

    def log(text: str) -> None:
        events.on_log(text)

    try:
        targets = load_targets(args.workspace / "configs")
        if args.target not in targets:
            raise TargetError(f"no remote target {args.target!r} in the workspace's configs/remotes.yaml")
        return dispatch(args.run_dir, targets[args.target], prepare_data=args.prepare_data, stdin=stream,
                        cancel_on_eof=args.cancel_on_eof, extra=tuple(extra), log=log)
    except (OSError, RuntimeError, TargetError, subprocess.SubprocessError) as exc:
        log(f"Remote run failed: {exc}")
        events.emit("run_end", status="failed", metrics={}, best_checkpoint=None, duration_s=0.0,
                    error=mask_text(str(exc))[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
