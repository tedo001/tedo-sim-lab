"""The lab's index of experiments, runs, metrics, artifacts, models and jobs.

Paths inside the workspace are stored relative to it (POSIX separators), so a
project folder keeps working after it is moved, synced or opened on another
machine; paths outside it are stored absolute.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from core.common.paths import AppPaths

from .database import Database, utc_now

__all__ = ["JobStatus", "LabStore", "RunStatus", "new_id"]

RunStatus = Literal["queued", "running", "completed", "early_stopped", "cancelled", "failed",
                    "interrupted"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
_FINISHED = ("completed", "early_stopped", "cancelled", "failed", "interrupted")


def new_id() -> str:
    """16 hex characters: unique enough for one lab, short enough for a folder name."""
    return uuid.uuid4().hex[:16]


class LabStore:
    def __init__(self, db: Database, workspace: Path) -> None:
        self.db = db
        self.workspace = workspace.resolve()

    @classmethod
    def open(cls, paths: AppPaths) -> LabStore:
        """Open (creating and migrating if needed) the workspace's ``database/lab.db``."""
        db = Database(paths.lab_db)
        db.migrate()
        return cls(db, paths.workspace)

    def close(self) -> None:
        self.db.close()

    # Paths -----------------------------------------------------------------
    def stored_path(self, path: Path | str) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.workspace).as_posix()
        except ValueError:
            return resolved.as_posix()

    def resolve_path(self, stored: str) -> Path:
        path = Path(stored)
        return path if path.is_absolute() else self.workspace / path

    # Experiments -----------------------------------------------------------
    def create_experiment(self, name: str, task: str, spec_yaml: str, spec_hash: str,
                          tags: Iterable[str] = ()) -> str:
        experiment_id = new_id()
        self.db.execute("INSERT INTO experiments (id, name, task, spec_yaml, spec_hash, tags, "
                        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (experiment_id, name, task, spec_yaml, spec_hash, json.dumps(list(tags)),
                         utc_now()))
        return experiment_id

    def experiment(self, experiment_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))

    def experiments(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))

    # Runs ------------------------------------------------------------------
    def create_run(self, experiment_id: str, run_dir: Path, runner: str, *,
                   run_id: str | None = None, parent_run_id: str | None = None) -> str:
        run_id = run_id or new_id()
        self.db.execute("INSERT INTO runs (id, experiment_id, parent_run_id, status, runner, run_dir, "
                        "created_at) VALUES (?, ?, ?, 'queued', ?, ?, ?)",
                        (run_id, experiment_id, parent_run_id, runner, self.stored_path(run_dir),
                         utc_now()))
        return run_id

    def run(self, run_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def run_dir(self, run_id: str) -> Path:
        row = self.run(run_id)
        if row is None:
            raise KeyError(f"no run {run_id!r}")
        return self.resolve_path(row["run_dir"])

    def runs(self, *, experiment_id: str | None = None, status: RunStatus | None = None,
             limit: int = 200) -> list[sqlite3.Row]:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.db.query(f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",
                             (*params, limit))

    def start_run(self, run_id: str, *, device: str | None = None,
                  git_commit: str | None = None) -> None:
        self._update_run(run_id, status="running", started_at=utc_now(), device=device,
                         git_commit=git_commit)

    def finish_run(self, run_id: str, status: RunStatus, *, error: str | None = None,
                   duration_s: float | None = None) -> None:
        if status not in _FINISHED:
            raise ValueError(f"{status!r} is not a finished status")
        self._update_run(run_id, status=status, ended_at=utc_now(), error=error,
                         duration_s=duration_s)

    def set_run_details(self, run_id: str, *, device: str | None = None,
                        git_commit: str | None = None) -> None:
        self._update_run(run_id, device=device, git_commit=git_commit)

    def requeue_run(self, run_id: str) -> None:
        """A finished run goes back in the queue (resume): no end time, no error."""
        cursor = self.db.execute("UPDATE runs SET status = 'queued', ended_at = NULL, error = NULL, "
                                 "duration_s = NULL WHERE id = ?", (run_id,))
        if cursor.rowcount == 0:
            raise KeyError(f"no run {run_id!r}")

    _RUN_VIEW = ("SELECT r.*, e.name AS experiment_name, e.task, e.spec_yaml FROM runs r "
                 "JOIN experiments e ON e.id = r.experiment_id")

    def runs_with_experiments(self, *, limit: int = 200) -> list[sqlite3.Row]:
        """Runs, newest first, with their experiment's name, task and spec."""
        return self.db.query(f"{self._RUN_VIEW} ORDER BY r.created_at DESC LIMIT ?", (limit,))

    def run_with_experiment(self, run_id: str) -> sqlite3.Row | None:
        return self.db.query_one(f"{self._RUN_VIEW} WHERE r.id = ?", (run_id,))

    def set_mlflow_run(self, run_id: str, mlflow_run_id: str) -> None:
        self._update_run(run_id, mlflow_run_id=mlflow_run_id)

    def _update_run(self, run_id: str, **values: Any) -> None:
        values = {key: value for key, value in values.items() if value is not None or key == "error"}
        assignments = ", ".join(f"{key} = ?" for key in values)
        cursor = self.db.execute(f"UPDATE runs SET {assignments} WHERE id = ?",
                                 (*values.values(), run_id))
        if cursor.rowcount == 0:
            raise KeyError(f"no run {run_id!r}")

    # Params and metrics -------------------------------------------------------
    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        with self.db.transaction() as connection:
            connection.executemany(
                "INSERT INTO params (run_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT (run_id, key) DO UPDATE SET value = excluded.value",
                [(run_id, key, json.dumps(value, default=str)) for key, value in params.items()])

    def params(self, run_id: str) -> dict[str, Any]:
        rows = self.db.query("SELECT key, value FROM params WHERE run_id = ? ORDER BY key", (run_id,))
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def log_metrics(self, run_id: str, metrics: Mapping[str, float], *, step: int,
                    epoch: int | None = None) -> None:
        now = utc_now()
        with self.db.transaction() as connection:
            connection.executemany(
                "INSERT INTO metrics (run_id, key, step, epoch, value, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (run_id, key, step) "
                "DO UPDATE SET value = excluded.value, epoch = excluded.epoch",
                [(run_id, key, step, epoch, float(value), now) for key, value in metrics.items()])

    def metric_history(self, run_id: str, key: str) -> list[tuple[int, float]]:
        rows = self.db.query("SELECT step, value FROM metrics WHERE run_id = ? AND key = ? "
                             "ORDER BY step", (run_id, key))
        return [(row["step"], row["value"]) for row in rows]

    def latest_metrics(self, run_id: str) -> dict[str, float]:
        rows = self.db.query("SELECT m.key, m.value FROM metrics m JOIN (SELECT key, MAX(step) AS "
                             "step FROM metrics WHERE run_id = ? GROUP BY key) last ON "
                             "m.key = last.key AND m.step = last.step WHERE m.run_id = ?",
                             (run_id, run_id))
        return {row["key"]: row["value"] for row in rows}

    # Artifacts, datasets, models, notebooks ------------------------------------
    def add_artifact(self, run_id: str, path: Path, kind: str, *, sha256: str | None = None,
                     size_bytes: int | None = None) -> None:
        self.db.execute("INSERT INTO artifacts (run_id, kind, path, sha256, size_bytes, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (run_id, path) DO UPDATE SET "
                        "kind = excluded.kind, sha256 = excluded.sha256, "
                        "size_bytes = excluded.size_bytes",
                        (run_id, kind, self.stored_path(path), sha256, size_bytes, utc_now()))

    def artifacts(self, run_id: str) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM artifacts WHERE run_id = ? ORDER BY id", (run_id,))

    def upsert_dataset(self, card_id: str, status: str, *, local_path: Path | None = None,
                       fingerprint: str | None = None) -> None:
        stored = self.stored_path(local_path) if local_path else None
        self.db.execute("INSERT INTO datasets (card_id, local_path, status, fingerprint, updated_at) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (card_id) DO UPDATE SET "
                        "local_path = excluded.local_path, status = excluded.status, "
                        "fingerprint = excluded.fingerprint, updated_at = excluded.updated_at",
                        (card_id, stored, status, fingerprint, utc_now()))

    def register_model(self, name: str, *, card_id: str | None = None,
                       source_run_id: str | None = None, checkpoint_path: Path | None = None,
                       metrics: Mapping[str, float] | None = None) -> tuple[str, int]:
        """Add the next version of ``name``; return ``(model_id, version)``."""
        with self.db.transaction() as connection:
            row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM models WHERE name = ?",
                                     (name,)).fetchone()
            version = int(row[0]) + 1
            model_id = new_id()
            connection.execute(
                "INSERT INTO models (id, name, version, card_id, source_run_id, checkpoint_path, "
                "metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model_id, name, version, card_id, source_run_id,
                 self.stored_path(checkpoint_path) if checkpoint_path else None,
                 json.dumps(dict(metrics or {})), utc_now()))
        return model_id, version

    def models(self, *, name: str | None = None) -> list[sqlite3.Row]:
        """Registered model versions, newest first (optionally one name)."""
        if name is not None:
            return self.db.query("SELECT * FROM models WHERE name = ? ORDER BY version DESC", (name,))
        return self.db.query("SELECT * FROM models ORDER BY created_at DESC, version DESC")

    def model(self, model_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))

    def update_model(self, model_id: str, *, stage: str | None = None, notes: str | None = None,
                     mlflow_version: str | None = None) -> None:
        """Change a version's stage ('none', 'staging', 'production', 'archived'), notes or MLflow
        version; at most one version of a name is in production."""
        with self.db.transaction() as connection:
            if stage == "production":
                name = connection.execute("SELECT name FROM models WHERE id = ?", (model_id,)).fetchone()[0]
                connection.execute("UPDATE models SET stage = 'archived' WHERE name = ? AND "
                                   "stage = 'production' AND id != ?", (name, model_id))
            for column, value in (("stage", stage), ("notes", notes), ("mlflow_version", mlflow_version)):
                if value is not None:
                    connection.execute(f"UPDATE models SET {column} = ? WHERE id = ?", (value, model_id))

    def delete_model(self, model_id: str) -> None:
        """Forget a registered version (its checkpoint stays in the run folder)."""
        self.db.execute("DELETE FROM models WHERE id = ?", (model_id,))

    def link_notebook(self, notebook_path: Path, *, experiment_id: str | None = None,
                      run_id: str | None = None) -> None:
        if not (experiment_id or run_id):
            raise ValueError("link a notebook to an experiment, a run, or both")
        self.db.execute("INSERT INTO notebook_links (notebook_path, experiment_id, run_id, created_at) "
                        "VALUES (?, ?, ?, ?)", (self.stored_path(notebook_path), experiment_id,
                                                run_id, utc_now()))

    def notebook_links(self) -> list[sqlite3.Row]:
        """Every notebook ↔ run link (paths as stored: relative to the workspace)."""
        return self.db.query("SELECT notebook_path, experiment_id, run_id FROM notebook_links "
                             "ORDER BY created_at")

    # Jobs ------------------------------------------------------------------
    def add_job(self, job_id: str, kind: Literal["run", "task"], title: str, *,
                run_id: str | None = None) -> None:
        self.db.execute("INSERT INTO jobs (id, kind, title, run_id, status, created_at) "
                        "VALUES (?, ?, ?, ?, 'queued', ?)", (job_id, kind, title, run_id, utc_now()))

    def set_job_status(self, job_id: str, status: JobStatus, *, error: str | None = None) -> None:
        column = "started_at" if status == "running" else "ended_at"
        self.db.execute(f"UPDATE jobs SET status = ?, {column} = ?, error = ? WHERE id = ?",
                        (status, utc_now(), error, job_id))

    def jobs(self, *, limit: int = 100) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def recover_interrupted(self) -> int:
        """At start-up: whatever was queued or running when the app last stopped is interrupted."""
        now = utc_now()
        with self.db.transaction() as connection:
            jobs = connection.execute("UPDATE jobs SET status = 'interrupted', ended_at = ? "
                                      "WHERE status IN ('queued', 'running')", (now,)).rowcount
            runs = connection.execute("UPDATE runs SET status = 'interrupted', ended_at = ? "
                                      "WHERE status IN ('queued', 'running')", (now,)).rowcount
        return jobs + runs

    def counts(self) -> dict[str, int]:
        return {table: int(self.db.query_one(f"SELECT COUNT(*) FROM {table}")[0])
                for table in ("experiments", "runs", "models", "jobs")}
