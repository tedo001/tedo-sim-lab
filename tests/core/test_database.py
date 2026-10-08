"""Schema, migrations and the LabStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.common import AppPaths
from core.tracking import Database, LabStore, MigrationError, discover_migrations
from core.tracking.database import MIGRATIONS_DIR

TABLES = {"experiments", "runs", "params", "metrics", "artifacts", "datasets", "models",
          "notebook_links", "jobs"}


@pytest.fixture
def store(workspace: AppPaths) -> LabStore:
    store = LabStore.open(workspace)
    yield store
    store.close()


def test_migrations_create_every_table(store: LabStore) -> None:
    assert store.db.version == len(discover_migrations()) == 1
    assert store.db.tables() == TABLES


def test_migrating_again_changes_nothing(workspace: AppPaths, store: LabStore) -> None:
    again = LabStore.open(workspace)
    assert again.db.version == 1
    again.close()


def test_newer_database_is_refused(workspace: AppPaths, store: LabStore) -> None:
    store.db.execute("PRAGMA user_version = 99")
    with pytest.raises(MigrationError, match="newer version"):
        Database(workspace.lab_db).migrate()


def test_failed_migration_rolls_back(tmp_path: Path) -> None:
    folder = tmp_path / "migrations"
    folder.mkdir()
    (folder / "0001_ok.sql").write_text("CREATE TABLE a (x INTEGER);")
    (folder / "0002_broken.sql").write_text("CREATE TABLE b (x INTEGER); NOT VALID SQL;")
    db = Database(tmp_path / "t.db", migrations_dir=folder)
    with pytest.raises(MigrationError, match="0002_broken.sql"):
        db.migrate()
    assert db.version == 1 and db.tables() == {"a"}


@pytest.mark.parametrize("names, message", [
    (["0001_a.sql", "0003_c.sql"], "without gaps"),
    (["1_a.sql"], "NNNN_lower_case"),
])
def test_migration_files_are_checked(tmp_path: Path, names: list[str], message: str) -> None:
    for name in names:
        (tmp_path / name).write_text("SELECT 1;")
    with pytest.raises(MigrationError, match=message):
        discover_migrations(tmp_path)


def test_shipped_migrations_are_well_formed() -> None:
    assert discover_migrations(MIGRATIONS_DIR)[0][1].name == "0001_initial.sql"


def test_experiment_and_run_lifecycle(workspace: AppPaths, store: LabStore) -> None:
    experiment = store.create_experiment("mnist", "image_classification", "name: mnist\n", "h1",
                                         ["demo"])
    run_dir = workspace.experiments / experiment / "runs" / "r1"
    run = store.create_run(experiment, run_dir, "torch_classification", run_id="r1")
    assert store.run(run)["status"] == "queued"
    assert store.run(run)["run_dir"] == f"experiments/{experiment}/runs/r1"  # relative
    assert store.run_dir(run) == run_dir.resolve()
    store.start_run(run, device="cpu", git_commit="abc123")
    store.log_params(run, {"lr": 0.01, "layers": [32, 64]})
    store.log_metrics(run, {"loss": 1.0, "accuracy": 0.4}, step=1, epoch=1)
    store.log_metrics(run, {"loss": 0.5}, step=2, epoch=2)
    store.log_metrics(run, {"loss": 0.45}, step=2, epoch=2)  # same step overwrites
    store.finish_run(run, "completed", duration_s=12.5)
    row = store.run(run)
    assert (row["status"], row["device"], row["duration_s"]) == ("completed", "cpu", 12.5)
    assert store.params(run) == {"layers": [32, 64], "lr": 0.01}
    assert store.metric_history(run, "loss") == [(1, 1.0), (2, 0.45)]
    assert store.latest_metrics(run) == {"accuracy": 0.4, "loss": 0.45}
    assert [r["id"] for r in store.runs(experiment_id=experiment, status="completed")] == [run]
    with pytest.raises(ValueError, match="not a finished status"):
        store.finish_run(run, "running")


def test_paths_outside_the_workspace_stay_absolute(store: LabStore, tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere" / "ckpt.pt"
    assert store.stored_path(outside) == outside.resolve().as_posix()
    assert store.resolve_path(store.stored_path(outside)) == outside.resolve()


def test_moved_workspace_still_finds_its_runs(tmp_path: Path) -> None:
    first = AppPaths.resolve(tmp_path / "a").ensure()
    store = LabStore.open(first)
    experiment = store.create_experiment("x", "image_classification", "", "h")
    run = store.create_run(experiment, first.experiments / "run", "r")
    store.close()
    (tmp_path / "a").rename(tmp_path / "b")
    moved = LabStore.open(AppPaths.resolve(tmp_path / "b"))
    assert moved.run_dir(run) == (tmp_path / "b" / "experiments" / "run").resolve()
    moved.close()


def test_foreign_keys_and_cascades(store: LabStore, workspace: AppPaths) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.create_run("missing-experiment", workspace.experiments / "x", "r")
    experiment = store.create_experiment("x", "image_classification", "", "h")
    run = store.create_run(experiment, workspace.experiments / "x", "r")
    store.log_metrics(run, {"loss": 1.0}, step=1)
    store.db.execute("DELETE FROM experiments WHERE id = ?", (experiment,))
    assert store.run(run) is None and store.metric_history(run, "loss") == []


def test_model_versions_and_artifacts(store: LabStore, workspace: AppPaths) -> None:
    experiment = store.create_experiment("x", "image_classification", "", "h")
    run = store.create_run(experiment, workspace.experiments / "x", "r")
    assert store.register_model("digits-cnn", source_run_id=run)[1] == 1
    assert store.register_model("digits-cnn", metrics={"acc": 0.99})[1] == 2
    store.add_artifact(run, workspace.experiments / "x" / "best.pt", "checkpoint", size_bytes=10)
    store.add_artifact(run, workspace.experiments / "x" / "best.pt", "checkpoint", size_bytes=20)
    artifacts = store.artifacts(run)
    assert len(artifacts) == 1 and artifacts[0]["size_bytes"] == 20
    store.link_notebook(workspace.notebooks / "a.ipynb", run_id=run)
    with pytest.raises(ValueError):
        store.link_notebook(workspace.notebooks / "a.ipynb")


def test_interrupted_work_is_recovered(store: LabStore, workspace: AppPaths) -> None:
    experiment = store.create_experiment("x", "image_classification", "", "h")
    running = store.create_run(experiment, workspace.experiments / "a", "r")
    store.start_run(running)
    done = store.create_run(experiment, workspace.experiments / "b", "r")
    store.finish_run(done, "completed")
    store.add_job("j1", "run", "training", run_id=running)
    store.set_job_status("j1", "running")
    assert store.recover_interrupted() == 2
    assert store.run(running)["status"] == "interrupted"
    assert store.run(done)["status"] == "completed"
    assert store.jobs()[0]["status"] == "interrupted"
