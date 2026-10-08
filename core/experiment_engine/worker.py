"""Run one experiment in its own process.

    python -m core.experiment_engine.worker <run_dir> [--runner module:Class] [--cancel-on-eof]
                                                       [--resume <checkpoint>]
                                                       [--evaluate <checkpoint> --split test|val]
    python -m core.experiment_engine.worker <folder> --benchmark [--cancel-on-eof]

Reads ``<run_dir>/experiment.yaml``, picks the runner, runs it, and writes
:class:`~core.experiment_engine.events.ProgressEvent` lines to stdout. Anything
else the process prints goes to stderr, so the event stream stays clean.

Cancel by writing ``cancel`` on stdin: the runner stops at its next check and
the process exits 2. With ``--cancel-on-eof`` (what the app uses) a closed stdin
also cancels, so a worker never outlives a crashed app.

Exit codes: 0 completed or early-stopped, 1 failed, 2 cancelled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from core.common.cancel import Cancelled, CancelToken
from core.common.logging_setup import setup_logging
from core.common.masking import mask_text
from core.common.paths import CODE_ROOT
from core.common.references import resolve
from core.hardware.devices import resolve_device

from .events import JsonLinesCallbacks
from .recording import RunRecording, open_recording
from .runner import RunContext, RunnerRegistry, RunResult
from .spec import SpecError, load_spec

__all__ = ["BENCHMARK_FILE", "EXIT_CANCELLED", "EXIT_FAILED", "EXIT_OK", "main", "run_benchmark_job",
           "run_experiment"]

#: What a benchmark folder holds: the request (``target`` plus its settings) and, after, the results.
BENCHMARK_FILE, BENCHMARK_RESULTS = "benchmark.json", "results.json"

EXIT_OK, EXIT_FAILED, EXIT_CANCELLED = 0, 1, 2
_EXIT = {"completed": EXIT_OK, "early_stopped": EXIT_OK, "cancelled": EXIT_CANCELLED,
         "failed": EXIT_FAILED}

log = logging.getLogger("tedo.worker")


def watch_for_cancel(stream: TextIO, token: CancelToken, *, cancel_on_eof: bool) -> threading.Thread:
    """Cancel ``token`` when a line ``cancel`` arrives (or at EOF, if asked)."""

    def watch() -> None:
        for line in stream:
            if line.strip().lower() == "cancel":
                token.cancel()
                return
        if cancel_on_eof:
            token.cancel()

    thread = threading.Thread(target=watch, name="cancel-watcher", daemon=True)
    thread.start()
    return thread


def detach_stdin() -> TextIO:
    """Hand the worker's stdin (the "cancel" channel) to a private file and point standard input at
    the null device. On Windows a thread blocked reading a pipe makes every child process that
    would inherit that pipe hang at start (``git`` for the snapshot, tools libraries launch), so
    children must never get it. Returns the stream the cancel watcher should read."""
    try:
        private = os.dup(sys.stdin.fileno())  # not inheritable (PEP 446)
    except (OSError, ValueError, AttributeError):
        return sys.stdin
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":  # make sure child processes see the null device as their stdin
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(0))  # STD_INPUT_HANDLE
    sys.stdin = open(os.devnull, encoding="utf-8")  # lives as long as the process
    return open(private, encoding="utf-8", errors="replace")


def run_experiment(run_dir: Path, callbacks: JsonLinesCallbacks, token: CancelToken, *,
                   runner_entry: str | None = None, resume_from: Path | None = None) -> RunResult:
    """Load the spec, choose the runner and run it; never raises."""
    started = time.monotonic()
    run_id = run_dir.name
    recording: RunRecording | None = None
    try:
        spec = load_spec(run_dir / "experiment.yaml")
        registry = RunnerRegistry()
        if runner_entry:
            problems = registry.load_entries([runner_entry])
            if problems:
                raise SpecError(problems[0])
        else:
            registry.load_config(CODE_ROOT / "configs" / "runners.yaml")
        runner = registry.for_spec(spec)
        problems = runner.validate(spec)
        if problems:
            raise SpecError("; ".join(problems))
        if resume_from is not None and not resume_from.is_file():
            raise SpecError(f"cannot resume: {resume_from} does not exist")
        device = resolve_device(spec.runtime.device)
        recording = open_recording(spec, run_dir, run_id, runner=runner.id, device=device)
        ctx = RunContext(run_id, run_dir, device, token, tracker=recording.tracker,
                         resume_from=resume_from)
        callbacks.emit("run_start", device=device, run_dir=str(run_dir), **recording.start_payload())
        for note in recording.notes:
            callbacks.on_log(note)
        log.info("Running %s with %s on %s", spec.name, runner.id, ctx.device)
        result = runner.run(spec, callbacks, ctx)
    except (Cancelled, KeyboardInterrupt):
        result = RunResult(run_id, "cancelled", duration_s=time.monotonic() - started)
    except Exception as exc:
        log.exception("Run %s failed", run_id)
        result = RunResult(run_id, "failed", duration_s=time.monotonic() - started,
                           error=mask_text(f"{type(exc).__name__}: {exc}"))
    if recording is not None:
        try:
            recording.finish(run_dir, result)
        except Exception:  # recording must never change the outcome
            log.exception("Could not finish recording run %s", run_id)
    return result


def evaluations_dir(run_dir: Path) -> Path:
    return run_dir / "evaluations"


def run_evaluation(run_dir: Path, callbacks: JsonLinesCallbacks, token: CancelToken, *, checkpoint: Path,
                   split: str, runner_entry: str | None = None) -> RunResult:
    """Score ``checkpoint`` on ``split`` and write ``evaluations/<split>-<checkpoint>.json``;
    the lab database and MLflow are left alone. Never raises."""
    started = time.monotonic()
    run_id = run_dir.name
    try:
        spec = load_spec(run_dir / "experiment.yaml")
        registry = RunnerRegistry()
        if runner_entry:
            registry.load_entries([runner_entry])
        else:
            registry.load_config(CODE_ROOT / "configs" / "runners.yaml")
        runner = registry.for_spec(spec)
        if not checkpoint.is_file():
            raise SpecError(f"no checkpoint at {checkpoint}")
        ctx = RunContext(run_id, run_dir, resolve_device(spec.runtime.device), token)
        callbacks.emit("run_start", device=ctx.device, run_dir=str(run_dir), evaluation=True)
        report = runner.evaluate(spec, callbacks, ctx, checkpoint, split)
        report = {"checkpoint": checkpoint.name, "split": split, "device": ctx.device,
                  "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"), **report}
        folder = evaluations_dir(run_dir)
        folder.mkdir(exist_ok=True)
        target = folder / f"{split}-{checkpoint.stem}.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        callbacks.on_artifact(target, "evaluation")
        metrics = {f"{split}_{key}": float(value) for key, value in dict(report.get("metrics", {})).items()}
        return RunResult(run_id, "completed", metrics, duration_s=time.monotonic() - started)
    except (Cancelled, KeyboardInterrupt):
        return RunResult(run_id, "cancelled", duration_s=time.monotonic() - started)
    except Exception as exc:
        log.exception("Evaluation of %s failed", run_id)
        return RunResult(run_id, "failed", duration_s=time.monotonic() - started,
                         error=mask_text(f"{type(exc).__name__}: {exc}"))


def _allowed_benchmarks() -> list[str]:
    import yaml

    data = yaml.safe_load((CODE_ROOT / "configs" / "runners.yaml").read_text(encoding="utf-8")) or {}
    return list(data.get("benchmarks", []))


def run_benchmark_job(folder: Path, callbacks: JsonLinesCallbacks, token: CancelToken) -> RunResult:
    """Run the benchmark ``folder/benchmark.json`` asks for and write ``folder/results.json``. The
    target must be listed under ``benchmarks:`` in ``configs/runners.yaml``. Never raises."""
    started = time.monotonic()
    try:
        config = json.loads((folder / BENCHMARK_FILE).read_text(encoding="utf-8"))
        target = config.pop("target", "labs.benchmark.latency:run_benchmark")
        if target not in _allowed_benchmarks():
            raise SpecError(f"{target} is not a benchmark listed in configs/runners.yaml")
        function = resolve(target)
        ctx = RunContext(folder.name, folder, resolve_device(config.get("device", "auto")), token)
        callbacks.emit("run_start", device=ctx.device, run_dir=str(folder), benchmark=True)
        results = function(config, callbacks, ctx)
        results["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        target_file = folder / BENCHMARK_RESULTS
        target_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        callbacks.on_artifact(target_file, "benchmark")
        best = max((row["throughput"] for row in results.get("rows", [])), default=0.0)
        return RunResult(folder.name, "completed", {"best_throughput": float(best)},
                         duration_s=time.monotonic() - started)
    except (Cancelled, KeyboardInterrupt):
        return RunResult(folder.name, "cancelled", duration_s=time.monotonic() - started)
    except Exception as exc:
        log.exception("Benchmark %s failed", folder.name)
        return RunResult(folder.name, "failed", duration_s=time.monotonic() - started,
                         error=mask_text(f"{type(exc).__name__}: {exc}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="core.experiment_engine.worker")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--runner", default=None, help="force a runner ('module:Class')")
    parser.add_argument("--cancel-on-eof", action="store_true",
                        help="cancel when stdin closes (the app sets this)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to continue from (e.g. <run_dir>/checkpoints/last.pt)")
    parser.add_argument("--evaluate", type=Path, default=None, metavar="CHECKPOINT",
                        help="score this checkpoint instead of training")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--benchmark", action="store_true",
                        help="run_dir is a benchmark folder: time inference instead of training")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    events_out = sys.stdout
    sys.stdout = sys.stderr  # print() from any library must not corrupt the event stream
    setup_logging(None, args.log_level, console=True)
    run_dir = args.run_dir.resolve()
    callbacks = JsonLinesCallbacks(events_out, run_dir.name)
    token = CancelToken()
    watch_for_cancel(detach_stdin(), token, cancel_on_eof=args.cancel_on_eof)

    if args.benchmark:
        result = run_benchmark_job(run_dir, callbacks, token)
    elif args.evaluate is not None:
        result = run_evaluation(run_dir, callbacks, token, checkpoint=args.evaluate.resolve(),
                                split=args.split, runner_entry=args.runner)
    else:
        result = run_experiment(run_dir, callbacks, token, runner_entry=args.runner,
                                resume_from=args.resume.resolve() if args.resume else None)
    callbacks.on_run_end(result)
    return _EXIT[result.status]


if __name__ == "__main__":
    sys.exit(main())
