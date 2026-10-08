"""What runs an experiment: the runner contract, the context it gets, what it returns.

Runners live in ``labs/`` (and plugins). ``core`` never imports them: the
registry loads the ``"module:Class"`` entries listed in ``configs/runners.yaml``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import yaml

from core.common.cancel import CancelToken
from core.common.references import UnresolvedReference, resolve
from core.common.vocab import Task

from .spec import ExperimentSpec

__all__ = ["ExperimentRunner", "ExperimentalRunner", "NoRunnerError", "RunCallbacks", "RunContext",
           "RunResult", "RunStatus", "RunnerRegistry", "Tracker"]

log = logging.getLogger("tedo.runners")

RunStatus = Literal["completed", "early_stopped", "cancelled", "failed"]


class Tracker:
    """Where a run's params, metrics and artifacts go. This base records nothing;
    the SQLite + MLflow + run-folder tracker arrives in build phase 5."""

    def log_params(self, params: Mapping[str, object]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], *, step: int,
                    epoch: int | None = None) -> None: ...
    def log_artifact(self, path: Path, kind: str) -> None: ...


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    #: Resolved device: "cpu", "cuda:0", "mps".
    device: str
    cancel: CancelToken = field(default_factory=CancelToken)
    tracker: Tracker = field(default_factory=Tracker)
    #: Checkpoint to resume from, if any.
    resume_from: Path | None = None


class RunCallbacks:
    """Progress hooks a runner calls. Every hook is a no-op here; override what you need."""

    def on_run_start(self, ctx: RunContext) -> None: ...
    def on_epoch_start(self, epoch: int, total: int) -> None: ...
    def on_step(self, step: int, total: int, metrics: Mapping[str, float]) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None: ...
    def on_log(self, line: str) -> None: ...
    def on_artifact(self, path: Path, kind: str) -> None: ...
    def on_run_end(self, result: RunResult) -> None: ...


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    #: Final values plus ``best_*`` entries.
    metrics: dict[str, float] = field(default_factory=dict)
    best_checkpoint: Path | None = None
    duration_s: float = 0.0
    #: Already masked.
    error: str | None = None


class ExperimentRunner(ABC):
    #: Stable identifier stored with every run: "torch_classification", "xgboost", ...
    id: ClassVar[str]
    title: ClassVar[str]
    maturity: ClassVar[Literal["stable", "experimental"]] = "stable"
    tasks: ClassVar[frozenset[Task]] = frozenset()

    def supports(self, spec: ExperimentSpec) -> bool:
        return spec.task in self.tasks

    def validate(self, spec: ExperimentSpec) -> list[str]:
        """Problems that would stop ``spec`` running, in words; ``[]`` = good to go."""
        return [] if self.supports(spec) else [f"{self.title} does not run {spec.task.value}"]

    @abstractmethod
    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult: ...

    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        """Score a saved checkpoint on ``split`` ("test" or "val"): metrics, and whatever else the
        runner can say (classes, confusion matrix). Runners that cannot, say so."""
        raise NotImplementedError(f"{self.title} cannot evaluate checkpoints")


class ExperimentalRunner(ExperimentRunner):
    """A task the lab lists but cannot run yet. Selecting it explains when it arrives."""

    maturity = "experimental"
    planned_for: ClassVar[str] = "a later release"

    def validate(self, spec: ExperimentSpec) -> list[str]:
        return [f"{self.title} is experimental and not runnable yet (planned for {self.planned_for})"]

    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        raise NotImplementedError(f"{self.title} is planned for {self.planned_for}; "
                                  "this build cannot run it")


class NoRunnerError(LookupError):
    """No registered runner handles the experiment's task."""


class RunnerRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, ExperimentRunner] = {}
        self.errors: list[str] = []

    def register(self, runner: type[ExperimentRunner] | ExperimentRunner) -> None:
        instance = runner() if isinstance(runner, type) else runner
        if instance.id in self._runners:
            raise ValueError(f"runner id {instance.id!r} is registered twice")
        self._runners[instance.id] = instance

    def load_entries(self, entries: Iterable[str]) -> list[str]:
        """Register each ``"module:Class"``; a missing one is recorded, not raised."""
        problems = []
        for entry in entries:
            try:
                runner = resolve(entry)
                if not (isinstance(runner, type) and issubclass(runner, ExperimentRunner)):
                    raise TypeError("not an ExperimentRunner subclass")
                self.register(runner)
            except (UnresolvedReference, TypeError, ValueError) as exc:
                problems.append(f"{entry}: {exc}")
        self.errors += problems
        for problem in problems:
            log.warning("Runner not loaded: %s", problem)
        return problems

    def load_config(self, path: Path) -> list[str]:
        """Read ``runners:`` entries from a YAML file (``configs/runners.yaml``)."""
        if not path.is_file():
            return []
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return self.load_entries(data.get("runners", []))

    def get(self, runner_id: str) -> ExperimentRunner:
        return self._runners[runner_id]

    def all(self) -> list[ExperimentRunner]:
        return sorted(self._runners.values(), key=lambda runner: runner.title.lower())

    def __len__(self) -> int:
        return len(self._runners)

    def for_spec(self, spec: ExperimentSpec) -> ExperimentRunner:
        """The runner for ``spec``'s task: a stable one if any, else an experimental stub."""
        matches = [runner for runner in self._runners.values() if runner.supports(spec)]
        matches.sort(key=lambda runner: runner.maturity != "stable")
        if not matches:
            raise NoRunnerError(f"no runner handles {spec.task.value!r} in this build")
        return matches[0]

    def task_maturity(self) -> dict[Task, str]:
        """For each task some runner handles: "stable" if any handles it fully, else "experimental"."""
        result: dict[Task, str] = {}
        for runner in self._runners.values():
            for task in runner.tasks:
                if result.get(task) != "stable":
                    result[task] = runner.maturity
        return result
