"""What a worker records around a run: the reproducibility snapshot, and MLflow (when it
is installed and switched on in settings). Used by :mod:`core.experiment_engine.worker`."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.common.config import ConfigError, load_config, mlflow_tracking_uri
from core.common.optional import is_installed
from core.common.paths import AppPaths
from core.tracking.mlflow_tracker import CompositeTracker, MlflowTracker, flatten

from .runner import RunResult, Tracker
from .snapshot import take_snapshot
from .spec import ExperimentSpec

__all__ = ["ARTIFACTS", "MLFLOW_SWITCH", "RunRecording", "open_recording"]

#: ``TEDO_LAB_MLFLOW=0`` switches MLflow off for this process and its workers (the test suite).
MLFLOW_SWITCH = "TEDO_LAB_MLFLOW"

log = logging.getLogger("tedo.worker")

#: Run-folder files copied to MLflow at the end (path, artifact folder). last.pt stays local.
ARTIFACTS = (("experiment.yaml", "spec"), ("snapshot.json", "spec"), ("code.diff", "spec"),
             ("run_info.json", "results"), ("metrics.jsonl", "results"), ("test_confusion.json", "results"),
             ("results.json", "results"), ("checkpoints/model.joblib", "model"),
             ("explainer.npz", "explainer"), ("explainer.json", "explainer"),
             ("checkpoints/best.pt", "checkpoints"))


@dataclass
class RunRecording:
    tracker: Tracker
    snapshot: dict[str, Any]
    mlflow: MlflowTracker | None = None
    notes: list[str] = field(default_factory=list)

    def start_payload(self) -> dict[str, Any]:
        return {"git_commit": self.snapshot.get("git", {}).get("commit"),
                "mlflow_run_id": self.mlflow.run_id if self.mlflow else None}

    def finish(self, run_dir: Path, result: RunResult) -> None:
        if result.metrics:
            self.tracker.log_metrics(result.metrics, step=int(result.metrics.get("epochs_completed", 0)))
        if self.mlflow is not None:
            for name, folder in ARTIFACTS:
                self.mlflow.log_artifact(run_dir / name, folder)
            self.mlflow.finish(result.status)


def open_recording(spec: ExperimentSpec, run_dir: Path, run_id: str, *, runner: str,
                   device: str) -> RunRecording:
    paths = AppPaths.resolve()
    snapshot = take_snapshot(run_dir, spec, code_root=paths.code_root, datasets_root=paths.datasets,
                             device=device, runner=runner)
    recording = RunRecording(Tracker(), snapshot)
    try:
        config = load_config(paths)
    except ConfigError as exc:
        recording.notes.append(f"MLflow not used: settings.yaml is invalid ({exc})")
        return recording
    if not config.mlflow_tracking or os.environ.get(MLFLOW_SWITCH) == "0":
        recording.notes.append("MLflow tracking is switched off")
        return recording
    if not is_installed("mlflow"):
        recording.notes.append("MLflow is not installed in the experiment Python; the run is recorded "
                               "in the lab database only")
        return recording
    mlflow = MlflowTracker(mlflow_tracking_uri(config, paths), paths.mlruns, experiment=spec.name)
    tags = {"tedo.run_id": run_id, "tedo.task": spec.task.value, "tedo.dataset": spec.data.dataset,
            "tedo.model": spec.model.model, "tedo.runner": runner}
    commit = snapshot.get("git", {}).get("commit")
    if commit:
        tags["mlflow.source.git.commit"] = commit
    if mlflow.start(run_id, tags) is None:
        recording.notes.append(f"MLflow tracking failed to start: {mlflow.problem}")
        return recording
    mlflow.log_params({**flatten(spec.model_dump(mode="json")), "snapshot.python": snapshot["python"],
                       "snapshot.device": device, **{f"snapshot.{k}": v for k, v
                                                     in snapshot["packages"].items() if v}})
    recording.mlflow = mlflow
    recording.tracker = CompositeTracker(mlflow)
    return recording
