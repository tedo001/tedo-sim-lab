"""Reproducibility snapshot and MLflow tracking around a run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.common.cancel import CancelToken
from core.common.paths import AppPaths
from core.experiment_engine.events import JsonLinesCallbacks, parse_line
from core.experiment_engine.snapshot import compare, dataset_fingerprint, environment, git_state
from core.experiment_engine.spec import dump_spec, load_spec_text
from core.experiment_engine.worker import run_experiment
from core.tracking.mlflow_tracker import MlflowTracker, flatten

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = """\
name: tracked
task: image_classification
data: {dataset: mnist}
model: {model: simple_cnn}
training: {kind: torch, epochs: 2}
runtime: {device: cpu}
"""


class Lines(JsonLinesCallbacks):
    def __init__(self) -> None:
        self.text: list[str] = []
        super().__init__(self, "run")  # type: ignore[arg-type]

    def write(self, text: str) -> None:
        self.text += [line for line in text.splitlines() if line]

    def flush(self) -> None:
        pass


def test_flatten_and_compare() -> None:
    flat = flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x" * 600})
    assert flat == {"a.b": "1", "a.c": "[1, 2]", "d": "x" * 500}
    now = environment(Path.cwd())
    assert now["python"] and "numpy" in now["packages"]
    old = {**now, "packages": {**now["packages"], "numpy": "0.0.1"}, "deterministic": True,
           "git": {**now["git"], "dirty": False}}
    notes = compare(old, now)
    assert any(note.startswith("numpy: 0.0.1 then") for note in notes) and len(notes) == 1
    assert "not deterministic" in compare({**now, "deterministic": False}, now)[-1]


def test_git_state_outside_a_checkout(tmp_path) -> None:
    assert git_state(tmp_path)["commit"] is None


def test_dataset_fingerprint_changes_with_the_files(tmp_path) -> None:
    folder = tmp_path / "data"
    assert dataset_fingerprint(folder) is None
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"12")
    first = dataset_fingerprint(folder)
    (folder / "b.bin").write_bytes(b"3")
    assert dataset_fingerprint(folder) != first


def test_mlflow_failure_does_not_break_the_run(tmp_path) -> None:
    tracker = MlflowTracker("unknown-scheme://nowhere", tmp_path / "mlruns", "x")
    assert tracker.start("r", {}) is None and tracker.problem
    tracker.log_metrics({"a": 1.0}, step=1)  # quietly ignored
    tracker.finish("completed")


def test_a_run_is_mirrored_to_mlflow(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    monkeypatch.delenv("TEDO_LAB_MLFLOW")
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    monkeypatch.syspath_prepend(str(FIXTURES))
    run_dir = paths.experiments / "e" / "run1"
    dump_spec(load_spec_text(SPEC), run_dir / "experiment.yaml")
    lines = Lines()
    result = run_experiment(run_dir, lines, CancelToken(), runner_entry="tedo_test_runners:CountingRunner")
    assert result.status == "completed"
    start = next(e for e in map(parse_line, lines.text) if e.type == "run_start")
    mlflow_run = start.payload["mlflow_run_id"]
    assert mlflow_run and "git_commit" in start.payload
    snapshot = json.loads((run_dir / "snapshot.json").read_text())
    assert snapshot["spec_hash"] and snapshot["dataset"]["id"] == "mnist" and snapshot["device"] == "cpu"

    from mlflow.tracking import MlflowClient
    client = MlflowClient(f"sqlite:///{paths.mlflow_db.as_posix()}")
    run = client.get_run(mlflow_run)
    assert run.info.status == "FINISHED" and run.data.tags["tedo.run_id"] == "run1"
    assert run.data.params["training.epochs"] == "2" and run.data.params["model.model"] == "simple_cnn"
    assert sorted({m.step for m in client.get_metric_history(mlflow_run, "loss")}) == [1, 2]
    artifacts = {item.path for item in client.list_artifacts(mlflow_run, "spec")}
    assert {"spec/experiment.yaml", "spec/snapshot.json"} <= artifacts
    assert Path(run.info.artifact_uri.removeprefix("file://")).is_relative_to(paths.mlruns)
