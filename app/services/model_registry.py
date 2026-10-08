"""The Model Registry: a finished run's checkpoint, registered under a name as a numbered
version with its metrics, a stage and notes. When the run was tracked in MLflow, the same
version is registered in MLflow's model registry too (in the background; failure only leaves
a note, the lab's own registry is the record)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.common.config import AppConfig, mlflow_tracking_uri
from core.common.optional import is_installed
from core.common.paths import AppPaths
from core.experiment_engine.recording import MLFLOW_SWITCH
from core.tracking import LabStore

from .experiments import ExperimentService
from .jobs import JobQueue

__all__ = ["CHECKPOINTS", "STAGES", "ModelRegistryService", "RegisteredModel"]

#: Checkpoints a run can register, best first: (file in the run folder, MLflow artifact folder).
CHECKPOINTS = (("checkpoints/best.pt", "checkpoints"), ("checkpoints/model.joblib", "model"),
               ("checkpoints/last.pt", None))
STAGES = ("none", "staging", "production", "archived")
#: Metrics worth keeping with a version.
_KEPT = ("test_", "cv_", "best_", "silhouette", "explained_variance", "adjusted_rand")


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    name: str
    version: int
    card_id: str | None
    source_run_id: str | None
    checkpoint: Path | None
    metrics: dict[str, float]
    created_at: str
    stage: str
    notes: str
    mlflow_version: str | None

    @property
    def label(self) -> str:
        return f"{self.name} v{self.version}"


class ModelRegistryService(QObject):
    changed = Signal()
    registered = Signal(str)  # model id of a new version
    #: model id, MLflow version or "", problem or "".
    mlflow_registered = Signal(str, str, str)

    def __init__(self, paths: AppPaths, config: AppConfig, store: LabStore, experiments: ExperimentService,
                 jobs: JobQueue, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config, self.store = paths, config, store
        self.experiments, self.jobs = experiments, jobs
        self._mlflow_jobs: dict[str, str] = {}  # job id → model id
        jobs.job_finished.connect(self._mlflow_done)

    # Registering ----------------------------------------------------------------------
    def checkpoint_of(self, run_id: str) -> tuple[Path, str | None] | None:
        """The run's best checkpoint on disk and its MLflow artifact folder, or ``None``."""
        run_dir = self.experiments.view(run_id).run_dir
        for name, folder in CHECKPOINTS:
            if (run_dir / name).is_file():
                return run_dir / name, folder
        return None

    def register(self, run_id: str, name: str | None = None) -> RegisteredModel:
        view = self.experiments.view(run_id)
        if not view.finished_ok:
            raise ValueError("only a completed run can be registered")
        found = self.checkpoint_of(run_id)
        if found is None:
            raise FileNotFoundError(f"run {run_id} has no checkpoint to register")
        checkpoint, folder = found
        metrics = {key: value for key, value in self.store.latest_metrics(run_id).items()
                   if key.startswith(_KEPT)}
        model_id, _version = self.store.register_model((name or view.name).strip() or view.name,
                                                       card_id=view.model, source_run_id=run_id,
                                                       checkpoint_path=checkpoint, metrics=metrics)
        model = self.get(model_id)
        if view.mlflow_run_id and folder and self.mlflow_enabled:
            self._register_in_mlflow(model, view.mlflow_run_id, folder)
        self.registered.emit(model.id)
        self.changed.emit()
        return model

    @property
    def mlflow_enabled(self) -> bool:
        return self.config.mlflow_tracking and os.environ.get(MLFLOW_SWITCH) != "0" and is_installed("mlflow")

    def _register_in_mlflow(self, model: RegisteredModel, mlflow_run_id: str, folder: str) -> None:
        uri = mlflow_tracking_uri(self.config, self.paths)

        def work(cancel: Any, progress: Any) -> str:
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=uri, registry_uri=uri)
            if not client.search_registered_models(filter_string=f"name = '{model.name}'"):
                client.create_registered_model(model.name, description="Registered from TEDO AI Research Lab")
            version = client.create_model_version(model.name, source=f"runs:/{mlflow_run_id}/{folder}",
                                                  run_id=mlflow_run_id, description=f"lab model {model.id}")
            return str(version.version)

        job_id = self.jobs.submit_task(work, title=f"Register {model.label} in MLflow", quiet=True)
        self._mlflow_jobs[job_id] = model.id

    def _mlflow_done(self, job_id: str, status: str) -> None:
        model_id = self._mlflow_jobs.pop(job_id, None)
        if model_id is None:
            return
        job = self.jobs.job(job_id)
        if status == "completed" and self.store.model(model_id) is not None:
            self.store.update_model(model_id, mlflow_version=str(job.result))
            self.mlflow_registered.emit(model_id, str(job.result), "")
            self.changed.emit()
        else:
            self.mlflow_registered.emit(model_id, "", job.error or status)

    # Reading and changing ------------------------------------------------------------
    def _from_row(self, row: Any) -> RegisteredModel:
        checkpoint = self.store.resolve_path(row["checkpoint_path"]) if row["checkpoint_path"] else None
        return RegisteredModel(row["id"], row["name"], int(row["version"]), row["card_id"],
                               row["source_run_id"], checkpoint, json.loads(row["metrics"] or "{}"),
                               row["created_at"], row["stage"], row["notes"], row["mlflow_version"])

    def all(self) -> list[RegisteredModel]:
        return [self._from_row(row) for row in self.store.models()]

    def get(self, model_id: str) -> RegisteredModel:
        row = self.store.model(model_id)
        if row is None:
            raise KeyError(model_id)
        return self._from_row(row)

    def set_stage(self, model_id: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {', '.join(STAGES)}")
        self.store.update_model(model_id, stage=stage)
        self.changed.emit()

    def set_notes(self, model_id: str, notes: str) -> None:
        self.store.update_model(model_id, notes=notes)
        self.changed.emit()

    def delete(self, model_id: str) -> None:
        self.store.delete_model(model_id)
        self.changed.emit()
