"""Experiments from the interface's side: create one from a spec, queue its run, record
what the worker reports in the lab database, cancel or resume it.

A run folder is ``<workspace>/experiments/<name>-<id>/<run id>/`` and holds
``experiment.yaml`` (what the worker reads), everything the runner writes, and
``run.log`` (every log line the worker printed).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.catalog import Catalog
from core.common import AppPaths
from core.experiment_engine.events import ProgressEvent
from core.experiment_engine.runner import NoRunnerError
from core.experiment_engine.snapshot import compare, environment
from core.experiment_engine.spec import ExperimentSpec, dump_spec, dump_spec_text, spec_hash
from core.tracking import LabStore, new_id

from .jobs import JobQueue
from .runs import FINISHED, LiveState, RunView, slug

__all__ = ["ExperimentService"]


class ExperimentService(QObject):
    run_changed = Signal(str)               # run id: created, started, finished
    run_epoch = Signal(str, int, object)    # run id, epoch, metrics
    run_step = Signal(str)                  # run id (read live(run_id))
    run_log = Signal(str, str)              # run id, line
    draft_changed = Signal(object)          # ExperimentSpec for the Experiment Builder
    explain_requested = Signal(str)         # run id for the CNN Explainer
    show_requested = Signal(str)            # run id for the Training page

    def __init__(self, paths: AppPaths, store: LabStore, jobs: JobQueue, catalog: Catalog,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.store, self.jobs, self.catalog = paths, store, jobs, catalog
        self._job_of_run: dict[str, str] = {}
        self._end: dict[str, Mapping[str, Any]] = {}
        self._live: dict[str, LiveState] = {}
        #: The spec waiting for the Experiment Builder (set by presets and "Open in builder").
        self.draft: ExperimentSpec | None = None
        #: The run the CNN Explainer should show next ("Open in CNN Explainer").
        self.explain_target: str | None = None
        jobs.job_started.connect(self._job_started)
        jobs.job_event.connect(self._job_event)
        jobs.job_finished.connect(self._job_finished)

    # Creating and controlling runs ---------------------------------------------
    def problems(self, spec: ExperimentSpec) -> list[str]:
        """What would stop ``spec`` from running, in words (checked in the app, before queueing)."""
        try:
            runner = self.catalog.runners.for_spec(spec)
        except NoRunnerError as exc:
            return [str(exc)]
        return runner.validate(spec)

    def launch(self, spec: ExperimentSpec) -> str:
        """Record the experiment, write its run folder and queue the run; return the run id."""
        runner = self.catalog.runners.for_spec(spec)
        experiment_id = self.store.create_experiment(spec.name, spec.task.value, dump_spec_text(spec),
                                                     spec_hash(spec), spec.tags)
        run_id = new_id()
        run_dir = self.paths.experiments / f"{slug(spec.name)}-{experiment_id[:6]}" / run_id
        dump_spec(spec, run_dir / "experiment.yaml")
        self.store.create_run(experiment_id, run_dir, runner.id, run_id=run_id)
        self._queue(run_id, run_dir, spec.name)
        return run_id

    def resume(self, run_id: str) -> None:
        view = self.view(run_id)
        if not view.resumable:
            raise ValueError("only a cancelled, failed or interrupted run with a last.pt can resume")
        self.store.requeue_run(run_id)
        self._queue(run_id, view.run_dir, view.name, ("--resume", str(view.last_checkpoint)))

    def reproduction_notes(self, run_id: str) -> list[str]:
        """What differs between the run's recorded environment and this one."""
        view = self.view(run_id)
        snapshot = view.run_dir / "snapshot.json"
        if not snapshot.is_file():
            return ["the original run has no snapshot.json (it predates phase 5), so differences "
                    "in the environment cannot be checked"]
        return compare(json.loads(snapshot.read_text(encoding="utf-8")), environment(self.paths.code_root))

    def reproduce(self, run_id: str) -> str:
        """Run the same experiment.yaml again as a child of ``run_id``; returns the new run id."""
        view = self.view(run_id)
        notes = self.reproduction_notes(run_id)
        new_id_ = new_id()
        run_dir = view.run_dir.parent / new_id_
        run_dir.mkdir(parents=True)
        shutil.copy2(view.run_dir / "experiment.yaml", run_dir / "experiment.yaml")
        (run_dir / "reproduces.json").write_text(json.dumps({"parent_run_id": run_id, "notes": notes},
                                                            indent=2), encoding="utf-8")
        runner = self.store.run(run_id)["runner"]
        self.store.create_run(view.experiment_id, run_dir, runner, run_id=new_id_, parent_run_id=run_id)
        self._queue(new_id_, run_dir, view.name)
        return new_id_

    def evaluate(self, run_id: str, checkpoint: str = "best", split: str = "test") -> str:
        """Queue scoring ``checkpoints/<checkpoint>.pt`` (``.joblib`` for scikit-learn) on ``split``;
        returns the job id. The result lands in ``evaluations/<split>-<checkpoint>.json``."""
        view = self.view(run_id)
        folder = view.run_dir / "checkpoints"
        path = next((folder / f"{checkpoint}{suffix}" for suffix in (".pt", ".joblib")
                     if (folder / f"{checkpoint}{suffix}").is_file()), None)
        if path is None:
            raise FileNotFoundError(f"run {run_id} has no {checkpoint} checkpoint")
        return self.jobs.submit_run(view.run_dir, title=f"Evaluate {view.name} ({checkpoint}, {split})",
                                    args=("--evaluate", str(path), "--split", split))

    def evaluations(self, run_id: str) -> list[tuple[str, Path]]:
        """(title, file) of every evaluation of a run: the test at the end of training first."""
        view = self.view(run_id)
        found = []
        if (view.run_dir / "test_confusion.json").is_file():
            title = ("End of fitting · test rows · fitted model" if view.tabular else
                     "End of training · test split · best checkpoint")
            found.append((title, view.run_dir / "test_confusion.json"))
        for path in sorted((view.run_dir / "evaluations").glob("*.json")):
            split, _, checkpoint = path.stem.partition("-")
            found.append((f"Re-evaluated · {'validation' if split == 'val' else split} split · {checkpoint} "
                          "checkpoint", path))
        return found

    def cancel(self, run_id: str) -> None:
        job_id = self._job_of_run.get(run_id)
        if job_id is not None:
            self.jobs.cancel(job_id)

    def _queue(self, run_id: str, run_dir: Path, title: str, args: tuple[str, ...] = ()) -> None:
        # The queue may start the job (and emit job_started) inside submit_run, so handlers find
        # the run through the job itself rather than through this mapping.
        job_id = self.jobs.submit_run(run_dir, title=title, run_id=run_id, args=args)
        self._job_of_run[run_id] = job_id
        self.run_changed.emit(run_id)

    def _run_for(self, job_id: str) -> str | None:
        job = self.jobs.job(job_id)
        return job.run_id if job.kind == "run" else None

    def open_in_builder(self, spec: ExperimentSpec, navigate: Any = None) -> None:
        self.draft = spec
        self.draft_changed.emit(spec)
        if navigate is not None:
            navigate("experiment_builder")

    def show(self, run_id: str, navigate: Any = None) -> None:
        """Open ``run_id`` on the Training page."""
        self.show_requested.emit(run_id)
        if navigate is not None:
            navigate("training")

    def explain(self, run_id: str, navigate: Any = None) -> None:
        self.explain_target = run_id
        self.explain_requested.emit(run_id)
        if navigate is not None:
            navigate("cnn_explainer")

    def explainable(self) -> list[RunView]:
        """Finished TinyVGG runs that exported weights for the CNN Explainer, newest first."""
        return [view for view in self.views(limit=200)
                if view.status in ("completed", "early_stopped")
                and (view.run_dir / "explainer.json").is_file()]

    # Reading ----------------------------------------------------------------------
    def views(self, limit: int = 100) -> list[RunView]:
        return [RunView.from_row(row, self.store.resolve_path(row["run_dir"]))
                for row in self.store.runs_with_experiments(limit=limit)]

    def view(self, run_id: str) -> RunView:
        row = self.store.run_with_experiment(run_id)
        if row is None:
            raise KeyError(run_id)
        return RunView.from_row(row, self.store.resolve_path(row["run_dir"]))

    def live(self, run_id: str) -> LiveState | None:
        return self._live.get(run_id)

    def history(self, run_id: str, keys: tuple[str, ...]) -> dict[str, list[tuple[int, float]]]:
        return {key: self.store.metric_history(run_id, key) for key in keys}

    def log_lines(self, run_id: str, limit: int = 400) -> list[str]:
        try:
            log = self.view(run_id).run_dir / "run.log"
            return log.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        except (KeyError, OSError):
            return []

    # Recording what the worker reports ---------------------------------------------
    def _job_started(self, job_id: str) -> None:
        run_id = self._run_for(job_id)
        if run_id is not None:
            self.store.start_run(run_id)
            self._live[run_id] = LiveState()
            self.run_changed.emit(run_id)

    def _job_event(self, job_id: str, event: ProgressEvent) -> None:
        run_id = self._run_for(job_id)
        if run_id is not None:
            self._record(run_id, event)

    def _record(self, run_id: str, event: ProgressEvent) -> None:
        payload, live = event.payload, self._live.setdefault(run_id, LiveState())
        if event.type == "run_start":
            self.store.set_run_details(run_id, device=payload.get("device"),
                                       git_commit=payload.get("git_commit"))
            if payload.get("mlflow_run_id"):
                self.store.set_mlflow_run(run_id, str(payload["mlflow_run_id"]))
            self.run_changed.emit(run_id)
        elif event.type == "epoch_start":
            live.on_epoch_start(int(payload.get("epoch", 0)), int(payload.get("total", 0)))
            self.run_step.emit(run_id)
        elif event.type == "step":
            live.on_step(int(payload.get("step", 0)), int(payload.get("total", 0)),
                         dict(payload.get("metrics") or {}))
            self.run_step.emit(run_id)
        elif event.type == "epoch_end":
            epoch, metrics = int(payload.get("epoch", 0)), dict(payload.get("metrics") or {})
            self.store.log_metrics(run_id, metrics, step=epoch, epoch=epoch)
            self.run_epoch.emit(run_id, epoch, metrics)
        elif event.type == "artifact" and payload.get("path"):
            path = Path(payload["path"])
            self.store.add_artifact(run_id, path, str(payload.get("kind", "file")),
                                    size_bytes=path.stat().st_size if path.is_file() else None)
        elif event.type == "run_end":
            self._end[run_id] = payload
        elif event.type == "log":
            self._log(run_id, str(payload.get("line", "")))

    def import_run(self, spec: ExperimentSpec, events: list[ProgressEvent],
                   fill: Callable[[Path], None]) -> str:
        """Record a run that happened elsewhere (Colab) from its folder's files and the worker's
        event stream, exactly as if it had run here; returns the new run id. ``fill(run_dir)``
        puts the files in place."""
        runner = self.catalog.runners.for_spec(spec)
        experiment_id = self.store.create_experiment(spec.name, spec.task.value, dump_spec_text(spec),
                                                     spec_hash(spec), spec.tags)
        run_id = new_id()
        run_dir = self.paths.experiments / f"{slug(spec.name)}-{experiment_id[:6]}" / run_id
        run_dir.mkdir(parents=True)
        fill(run_dir)
        dump_spec(spec, run_dir / "experiment.yaml")
        self.store.create_run(experiment_id, run_dir, runner.id, run_id=run_id)
        self.store.start_run(run_id)
        for event in events:
            if event.type != "artifact":  # artifact paths point into the other machine
                self._record(run_id, event)
        self._close_run(run_id, "failed", None)  # run_end, when present, decides the status
        return run_id

    def _log(self, run_id: str, line: str) -> None:
        try:
            run_dir = self.store.run_dir(run_id)
            with (run_dir / "run.log").open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except (KeyError, OSError):
            pass
        self.run_log.emit(run_id, line)

    def _job_finished(self, job_id: str, status: str) -> None:
        run_id = self._run_for(job_id)
        if run_id is None:
            return
        self._job_of_run.pop(run_id, None)
        self._close_run(run_id, status, self.jobs.job(job_id).error)

    def _close_run(self, run_id: str, status: str, job_error: str | None) -> None:
        end = self._end.pop(run_id, {})
        final = str(end.get("status") or status)
        if final not in FINISHED:
            final = "failed"
        metrics = {key: value for key, value in (end.get("metrics") or {}).items()
                   if isinstance(value, int | float)}
        if metrics:
            step = int(metrics.get("epochs_completed", 0))
            self.store.log_metrics(run_id, metrics, step=step, epoch=step or None)
        error = job_error or end.get("error")
        self.store.finish_run(run_id, final, error=error, duration_s=end.get("duration_s"))
        self._live.pop(run_id, None)
        self.run_changed.emit(run_id)
