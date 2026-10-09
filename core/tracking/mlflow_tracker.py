"""Mirror a run to MLflow (one MLflow run per lab run).

Tracking must never break training: if MLflow is missing, misconfigured or fails
part-way, the tracker logs a warning, stops talking to MLflow, and the run goes on.
Artifacts go to the workspace's ``mlruns/`` folder, never next to the code.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.experiment_engine.runner import Tracker

__all__ = ["CompositeTracker", "MlflowTracker", "flatten", "release_mlflow_stores"]

log = logging.getLogger("tedo.tracking")
_MAX_PARAM = 500  # MLflow's limit on a parameter value


def release_mlflow_stores() -> None:
    """Close the SQLite connections MLflow keeps open in this process (it caches one engine per
    database for the life of the process), so a workspace can be closed, moved or deleted; on
    Windows an open file cannot be removed. Does nothing when MLflow was never imported."""
    import sys

    if "mlflow" not in sys.modules:
        return
    for module_name in ("mlflow.store.tracking.sqlalchemy_store",
                        "mlflow.store.model_registry.sqlalchemy_store"):
        store = getattr(sys.modules.get(module_name), "SqlAlchemyStore", None)
        engines = getattr(store, "_engine_map", None)
        if not isinstance(engines, dict):
            continue
        for engine in list(engines.values()):
            try:
                engine.dispose()
            except Exception as exc:  # releasing is best effort; never fail closing the app
                log.debug("Could not dispose an MLflow engine: %s", exc)
        engines.clear()
    utils = sys.modules.get("mlflow.tracking._tracking_service.utils")
    for name in dir(utils) if utils else ():
        cache_clear = getattr(getattr(utils, name), "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested mappings → ``{"training.optimizer.lr": "0.001"}`` (lists as their text)."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:_MAX_PARAM]
    return flat


class CompositeTracker(Tracker):
    """Sends everything to several trackers."""

    def __init__(self, *trackers: Tracker) -> None:
        self.trackers = list(trackers)

    def log_params(self, params: Mapping[str, object]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step, epoch=epoch)

    def log_artifact(self, path: Path, kind: str) -> None:
        for tracker in self.trackers:
            tracker.log_artifact(path, kind)


class MlflowTracker(Tracker):
    def __init__(self, tracking_uri: str, artifact_root: Path, experiment: str) -> None:
        self.tracking_uri, self.artifact_root, self.experiment = tracking_uri, artifact_root, experiment
        self.client: Any = None
        self.run_id: str | None = None
        self.problem: str | None = None

    def start(self, run_name: str, tags: Mapping[str, str]) -> str | None:
        """Create the MLflow run; returns its id, or ``None`` (and sets :attr:`problem`)."""
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        try:
            from mlflow.tracking import MlflowClient

            self.client = MlflowClient(self.tracking_uri)
            found = self.client.get_experiment_by_name(self.experiment)
            if found is None:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
                experiment_id = self.client.create_experiment(self.experiment,
                                                              artifact_location=self.artifact_root.as_uri())
            else:
                experiment_id = found.experiment_id
            run = self.client.create_run(experiment_id, run_name=run_name, tags=dict(tags))
            self.run_id = run.info.run_id
        except Exception as exc:  # any MLflow failure: carry on without it
            self._give_up(exc)
        return self.run_id

    def _give_up(self, exc: Exception) -> None:
        self.problem = f"{type(exc).__name__}: {exc}"
        log.warning("MLflow tracking stopped: %s", self.problem)
        self.run_id = None

    def log_params(self, params: Mapping[str, object]) -> None:
        if self.run_id is None:
            return
        try:
            from mlflow.entities import Param

            flat = flatten(params)
            self.client.log_batch(self.run_id, params=[Param(k, v) for k, v in flat.items()])
        except Exception as exc:
            self._give_up(exc)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        if self.run_id is None:
            return
        try:
            from mlflow.entities import Metric

            now = int(time.time() * 1000)
            batch = [Metric(key, float(value), now, step) for key, value in metrics.items()
                     if isinstance(value, int | float)]
            self.client.log_batch(self.run_id, metrics=batch)
        except Exception as exc:
            self._give_up(exc)

    def log_artifact(self, path: Path, kind: str) -> None:
        if self.run_id is None or not path.is_file():
            return
        try:
            self.client.log_artifact(self.run_id, str(path), artifact_path=kind)
        except Exception as exc:
            self._give_up(exc)

    def finish(self, status: str) -> None:
        if self.run_id is None:
            return
        states = {"completed": "FINISHED", "early_stopped": "FINISHED", "cancelled": "KILLED"}
        state = states.get(status, "FAILED")
        try:
            self.client.set_terminated(self.run_id, status=state)
        except Exception as exc:
            self._give_up(exc)
