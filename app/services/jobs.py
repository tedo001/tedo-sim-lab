"""Background work without freezing the window.

* **Runs** (training, evaluation) go to a worker *process* through ``QProcess``:
  first in, first out, ``max_concurrent_runs`` at a time (1 by default: one GPU).
  Progress arrives as JSON lines and is re-emitted as :attr:`JobQueue.job_event`.
* **Tasks** (downloads, scans, connection tests) run on a ``QThreadPool``.

Cancel is cooperative first (``cancel`` on the worker's stdin, or the task's
:class:`CancelToken`), then forceful after a grace period.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThreadPool, QTimer, Signal

from core.common.cancel import CancelToken, ProgressFn
from core.common.masking import mask_text
from core.common.paths import WORKSPACE_ENV
from core.experiment_engine.events import ProgressEvent, parse_line
from core.tracking import LabStore, new_id

from .tasks import TaskRunnable

__all__ = ["Job", "JobQueue", "WorkerCommand", "worker_command"]

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
_EXIT_STATUS: dict[int, JobStatus] = {0: "completed", 2: "cancelled"}


@dataclass(frozen=True)
class WorkerCommand:
    program: str
    arguments: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


def worker_command(run_dir: Path, *, python: str, code_root: Path, workspace: Path | None = None,
                   extra: Sequence[str] = ()) -> WorkerCommand:
    """``python -m core.experiment_engine.worker <run_dir> --cancel-on-eof``, able to import the
    lab's code even when ``python`` is a separate experiment environment, and pointed at the
    same workspace (datasets, model downloads) as the app."""
    pythonpath = os.pathsep.join(filter(None, [str(code_root), os.environ.get("PYTHONPATH", "")]))
    env = {"PYTHONPATH": pythonpath, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if workspace is not None:
        env[WORKSPACE_ENV] = str(workspace)
    return WorkerCommand(python, ["-m", "core.experiment_engine.worker", str(run_dir),
                                  "--cancel-on-eof", *extra], code_root, env)


@dataclass
class Job:
    id: str
    kind: Literal["run", "task"]
    title: str
    status: JobStatus = "queued"
    run_dir: Path | None = None
    run_id: str | None = None
    #: Extra worker arguments, e.g. ``("--resume", "<checkpoint>")``.
    args: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    error: str | None = None
    result: Any = None
    cancel_requested: bool = False


class JobQueue(QObject):
    job_queued = Signal(str)
    job_started = Signal(str)
    job_event = Signal(str, object)          # job id, ProgressEvent
    job_progress = Signal(str, float, str)   # job id, fraction (<0 = unknown), message
    job_finished = Signal(str, str)          # job id, final status

    def __init__(self, command_factory: Callable[[Path], WorkerCommand], *,
                 max_concurrent_runs: int = 1, store: LabStore | None = None,
                 cancel_grace_ms: int = 10_000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._command_factory = command_factory
        self.max_concurrent_runs = max_concurrent_runs
        self._store = store
        self._grace_ms = cancel_grace_ms
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._processes: dict[str, QProcess] = {}
        self._retired: list[QProcess] = []
        self._buffers: dict[str, dict[str, str]] = {}
        self._last_run_end: dict[str, ProgressEvent] = {}
        self._tasks: dict[str, tuple[TaskRunnable, CancelToken]] = {}
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(4)

    # Inspection -------------------------------------------------------------
    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at)

    def job(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def active(self) -> list[Job]:
        return [job for job in self.jobs() if job.status in ("queued", "running")]

    # Runs -------------------------------------------------------------------
    def submit_run(self, run_dir: Path, *, title: str | None = None, run_id: str | None = None,
                   args: Sequence[str] = ()) -> str:
        job = Job(new_id(), "run", title or run_dir.name, run_dir=run_dir, run_id=run_id,
                  args=tuple(args))
        self._register(job)
        self._pending.append(job.id)
        self._pump()
        return job.id

    def _pump(self) -> None:
        while self._pending and len(self._processes) < self.max_concurrent_runs:
            self._start_process(self._jobs[self._pending.popleft()])

    def _start_process(self, job: Job) -> None:
        assert job.run_dir is not None
        command = self._command_factory(job.run_dir)
        process = QProcess()  # owned by Python, released after its signals finish (see _release)
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in command.env.items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(command.cwd))
        process.setProgram(command.program)
        process.setArguments([*command.arguments, *job.args])
        self._processes[job.id] = process
        self._buffers[job.id] = {"stdout": "", "stderr": ""}
        process.readyReadStandardOutput.connect(lambda: self._read(job.id, "stdout"))
        process.readyReadStandardError.connect(lambda: self._read(job.id, "stderr"))
        process.finished.connect(lambda code, exit_status: self._process_finished(
            job.id, code, exit_status == QProcess.ExitStatus.CrashExit))
        process.errorOccurred.connect(lambda error: self._process_error(job.id, error))
        self._set_status(job, "running")
        self.job_started.emit(job.id)
        process.start()

    def _read(self, job_id: str, stream: str, *, flush: bool = False) -> None:
        process = self._processes.get(job_id)
        if process is None:
            return
        raw = (process.readAllStandardOutput() if stream == "stdout"
               else process.readAllStandardError())
        text = self._buffers[job_id][stream] + bytes(raw.data()).decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._buffers[job_id][stream] = "" if flush else lines.pop()
        job = self._jobs[job_id]
        for line in lines:
            if not line.strip():
                continue
            event = parse_line(line, job.run_id or "", stream=stream)
            if event.type == "run_end":
                self._last_run_end[job_id] = event
            self.job_event.emit(job_id, event)

    def _process_finished(self, job_id: str, exit_code: int, crashed: bool) -> None:
        if job_id not in self._processes:
            return
        self._read(job_id, "stdout", flush=True)
        self._read(job_id, "stderr", flush=True)
        job = self._jobs[job_id]
        status = "failed" if crashed else _EXIT_STATUS.get(exit_code, "failed")
        if job.cancel_requested and status == "failed":
            status = "cancelled"  # killed after the grace period
        end = self._last_run_end.pop(job_id, None)
        if status == "failed":
            job.error = ((end.payload.get("error") if end else None)
                         or ("worker crashed" if crashed else f"worker exited with {exit_code}"))
        self._release(job_id)
        self._finish(job, status)
        self._pump()

    def _process_error(self, job_id: str, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and job_id in self._processes:
            job = self._jobs[job_id]
            job.error = f"could not start the worker: {self._processes[job_id].errorString()}"
            self._release(job_id)
            self._finish(job, "failed")
            self._pump()

    def _release(self, job_id: str) -> None:
        """Forget a finished process. It is still inside its own ``finished`` signal here, so it
        is kept alive until the event loop is back, then dropped (PySide6 owns it; no parent,
        no ``deleteLater``: deleting a QProcess mid-emission corrupts the heap)."""
        process = self._processes.pop(job_id, None)
        self._buffers.pop(job_id, None)
        if process is not None:
            self._retired.append(process)
            QTimer.singleShot(0, self._drop_retired)

    def _drop_retired(self) -> None:
        self._retired = [p for p in self._retired if p.state() != QProcess.ProcessState.NotRunning]

    # Tasks ------------------------------------------------------------------
    def submit_task(self, fn: Callable[[CancelToken, ProgressFn], Any], *, title: str) -> str:
        """Run ``fn(cancel_token, progress)`` on a worker thread; its return value becomes
        ``job(id).result``."""
        job = Job(new_id(), "task", title)
        self._register(job)
        token = CancelToken()
        runnable = TaskRunnable(fn, token)
        signals = runnable.signals
        signals.progress.connect(lambda fraction, message: self.job_progress.emit(job.id, fraction,
                                                                                  message))
        signals.succeeded.connect(lambda result: self._task_done(job.id, "completed", result=result))
        signals.failed.connect(lambda message: self._task_done(job.id, "failed", error=message))
        signals.cancelled.connect(lambda: self._task_done(job.id, "cancelled"))
        self._tasks[job.id] = (runnable, token)
        self._set_status(job, "running")
        self.job_started.emit(job.id)
        self._pool.start(runnable)
        return job.id

    def _task_done(self, job_id: str, status: JobStatus, *, result: Any = None,
                   error: str | None = None) -> None:
        job = self._jobs[job_id]
        job.result, job.error = result, error
        self._tasks.pop(job_id, None)
        self._finish(job, status)

    # Cancel and shutdown ------------------------------------------------------
    def cancel(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if job.status not in ("queued", "running"):
            return
        job.cancel_requested = True
        if job_id in self._pending:
            self._pending.remove(job_id)
            self._finish(job, "cancelled")
        elif job_id in self._processes:
            process = self._processes[job_id]
            process.write(b"cancel\n")
            QTimer.singleShot(self._grace_ms, lambda: self._kill_if_running(job_id))
        elif job_id in self._tasks:
            self._tasks[job_id][1].cancel()

    def _kill_if_running(self, job_id: str) -> None:
        process = self._processes.get(job_id)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def shutdown(self, timeout_ms: int = 5_000) -> None:
        """Cancel everything and wait (briefly) for workers to stop; for app exit."""
        for job_id in list(self._pending):
            self.cancel(job_id)
        for job_id, process in list(self._processes.items()):
            self._jobs[job_id].cancel_requested = True
            process.write(b"cancel\n")
            if not process.waitForFinished(timeout_ms):
                process.kill()
                process.waitForFinished(1_000)
        for _runnable, token in self._tasks.values():
            token.cancel()
        self._pool.waitForDone(timeout_ms)

    # Bookkeeping --------------------------------------------------------------
    def _register(self, job: Job) -> None:
        self._jobs[job.id] = job
        if self._store is not None:
            self._store.add_job(job.id, job.kind, job.title, run_id=job.run_id)
        self.job_queued.emit(job.id)

    def _set_status(self, job: Job, status: JobStatus) -> None:
        job.status = status
        if self._store is not None:
            self._store.set_job_status(job.id, status, error=mask_text(job.error) if job.error else None)

    def _finish(self, job: Job, status: JobStatus) -> None:
        if job.error:
            job.error = mask_text(job.error)
        self._set_status(job, status)
        self.job_finished.emit(job.id, status)
