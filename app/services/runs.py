"""What the interface needs to know about a run: its row in the lab database read into a
small object, and the live state of a running one (epoch, step, time left)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.experiment_engine.spec import ExperimentSpec, SpecError, load_spec_text

__all__ = ["FINISHED", "LiveState", "RunView", "slug"]

FINISHED = ("completed", "early_stopped", "cancelled", "failed", "interrupted")
#: Runs that can continue from their last checkpoint.
RESUMABLE = ("cancelled", "interrupted", "failed")


def slug(name: str, limit: int = 40) -> str:
    """A folder-safe version of an experiment name."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    return "-".join(part for part in cleaned.split("-") if part)[:limit] or "experiment"


@dataclass(frozen=True)
class RunView:
    id: str
    experiment_id: str
    name: str
    task: str
    status: str
    run_dir: Path
    device: str | None
    created_at: str
    started_at: str | None
    ended_at: str | None
    duration_s: float | None
    error: str | None
    spec: ExperimentSpec | None
    parent_run_id: str | None = None
    mlflow_run_id: str | None = None
    git_commit: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row, run_dir: Path) -> RunView:
        try:
            spec = load_spec_text(row["spec_yaml"])
        except SpecError:
            spec = None
        return cls(row["id"], row["experiment_id"], row["experiment_name"], row["task"], row["status"],
                   run_dir, row["device"], row["created_at"], row["started_at"], row["ended_at"],
                   row["duration_s"], row["error"], spec, row["parent_run_id"], row["mlflow_run_id"],
                   row["git_commit"])

    @property
    def dataset(self) -> str:
        return self.spec.data.dataset if self.spec else "?"

    @property
    def model(self) -> str:
        return self.spec.model.model if self.spec else "?"

    @property
    def epochs(self) -> int | None:
        return getattr(self.spec.training, "epochs", None) if self.spec else None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    @property
    def last_checkpoint(self) -> Path:
        return self.run_dir / "checkpoints" / "last.pt"

    @property
    def finished_ok(self) -> bool:
        return self.status in ("completed", "early_stopped")

    @property
    def resumable(self) -> bool:
        return self.status in RESUMABLE and self.last_checkpoint.is_file()


@dataclass
class LiveState:
    """Progress of a running run, from its events."""

    epoch: int = 0
    epochs: int = 0
    step: int = 0
    steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    #: Seconds per training step, measured over the run so far.
    _step_times: list[float] = field(default_factory=list)
    _last_step_at: float | None = None

    def on_epoch_start(self, epoch: int, epochs: int) -> None:
        self.epoch, self.epochs, self.step = epoch, epochs, 0
        self._last_step_at = time.monotonic()

    def on_step(self, step: int, steps: int, metrics: dict[str, float]) -> None:
        now = time.monotonic()
        if self._last_step_at is not None and step > self.step:
            self._step_times.append((now - self._last_step_at) / (step - self.step))
            self._step_times = self._step_times[-50:]
        self._last_step_at = now
        self.step, self.steps, self.metrics = step, steps, dict(metrics)

    @property
    def fraction(self) -> float:
        if not self.epochs:
            return 0.0
        within = self.step / self.steps if self.steps else 0.0
        return min(((self.epoch - 1) + within) / self.epochs, 1.0) if self.epoch else 0.0

    @property
    def eta_s(self) -> float | None:
        """Training time left (validation and testing not included), once steps were timed."""
        if not self._step_times or not self.steps or not self.epochs:
            return None
        per_step = sum(self._step_times) / len(self._step_times)
        remaining = (self.steps - self.step) + (self.epochs - self.epoch) * self.steps
        return max(remaining, 0) * per_step
