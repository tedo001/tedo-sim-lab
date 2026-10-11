"""Setting up the experiment Python from the app: check one, or make the per-user environment
(``core.common.experiment_env``) in a background task with pip's output streamed as progress.
Cancel stops the running command; a half-made environment is simply finished next time."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.common.cancel import Cancelled, CancelToken, ProgressFn
from core.common.experiment_env import (
    PythonFacts,
    Step,
    default_env_dir,
    describe_python,
    env_python,
    find_base_python,
    setup_steps,
    torch_build,
)
from core.common.masking import mask_text

from .jobs import JobQueue

__all__ = ["ExperimentSetup", "SetupPlan", "run_steps"]


class SetupPlan:
    """What making the environment would do here: from which Python, where, which PyTorch."""

    def __init__(self, *, cpu_only: bool = False, env_dir: Path | None = None,
                 base: list[str] | None = None, build: str | None = None) -> None:
        self.env_dir = env_dir or default_env_dir()
        self.base = base if base is not None else find_base_python()
        self.build = build or torch_build(cpu_only=cpu_only)
        self.steps: list[Step] = setup_steps(self.env_dir, self.base, self.build) if self.base else []

    @property
    def python(self) -> Path:
        return env_python(self.env_dir)


def run_steps(steps: list[Step], cancel: CancelToken, progress: ProgressFn) -> None:
    """Run each step, streaming its output; a failed step (after its fallback) stops with the
    last lines of output. Blocking: call from a task."""
    for index, step in enumerate(steps):
        progress(index / len(steps), step.title)
        for attempt, command in enumerate(filter(None, (step.command, step.fallback))):
            tail = _run(command, cancel, lambda line, i=index, t=step.title: progress(i / len(steps),
                                                                                   f"{t}: {line}"))
            if tail is None:
                break
            if attempt == 1 or step.fallback is None:
                raise RuntimeError(mask_text(f"{step.title} failed:\n" + "\n".join(tail)))
    progress(1.0, "Done")


def _run(command: list[str], cancel: CancelToken, line_out) -> list[str] | None:  # noqa: ANN001
    """``None`` when ``command`` succeeded, else its last lines of output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        if cancel.cancelled:
            process.terminate()
            process.wait(10)
            raise Cancelled()
        line = line.rstrip()
        if line:
            tail = [*tail[-14:], line]
            line_out(line[:160])
    return None if process.wait() == 0 else tail


class ExperimentSetup(QObject):
    #: job id, facts (PythonFacts) — a check finished
    checked = Signal(str, object)
    #: job id, python path ("" if it failed), error — making the environment finished
    made = Signal(str, str, str)

    def __init__(self, jobs: JobQueue, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self._checks: set[str] = set()
        self._makes: dict[str, Path] = {}
        jobs.job_finished.connect(self._finished)

    def check(self, python: str) -> str:
        job_id = self.jobs.submit_task(lambda cancel, progress: describe_python(python),
                                       title="Check the experiment Python", quiet=True)
        self._checks.add(job_id)
        return job_id

    def make(self, plan: SetupPlan) -> str:
        if not plan.steps:
            raise RuntimeError("No Python 3.11 or newer was found on this computer to make the "
                               "environment from. Install it from python.org, then try again.")

        def work(cancel: CancelToken, progress: ProgressFn) -> PythonFacts:
            run_steps(plan.steps, cancel, progress)
            facts = describe_python(str(plan.python))
            if not facts.ok or facts.missing():
                raise RuntimeError(f"The environment was made but is incomplete: "
                                   f"{facts.error or ', '.join(facts.missing())}")
            return facts

        job_id = self.jobs.submit_task(work, title="Set up the experiment Python")
        self._makes[job_id] = plan.python
        return job_id

    def _finished(self, job_id: str, status: str) -> None:
        job = self.jobs.job(job_id)
        if job_id in self._checks:
            self._checks.discard(job_id)
            result = job.result if status == "completed" else PythonFacts("", error=job.error or status)
            self.checked.emit(job_id, result)
        elif job_id in self._makes:
            python = self._makes.pop(job_id)
            ok = status == "completed"
            self.made.emit(job_id, str(python) if ok else "", "" if ok else (job.error or status))
