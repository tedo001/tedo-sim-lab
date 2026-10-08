"""Inference benchmarks through the worker: an untrained image model, a finished scikit-learn
run, and the checks that refuse a bad request."""

from __future__ import annotations

import json

import pytest

from core.common.cancel import CancelToken
from core.common.paths import AppPaths
from core.experiment_engine.worker import BENCHMARK_FILE, run_benchmark_job
from tests.labs.test_classical_ml import Recorder, fit, make_spec


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def bench(paths: AppPaths, request: dict, name: str = "bench"):
    folder = paths.results / "benchmarks" / name
    folder.mkdir(parents=True)
    (folder / BENCHMARK_FILE).write_text(json.dumps(request), encoding="utf-8")
    recorder = Recorder()
    result = run_benchmark_job(folder, recorder, CancelToken())
    return folder, recorder, result


def test_untrained_image_model(workspace) -> None:
    pytest.importorskip("torch")
    folder, recorder, result = bench(workspace, {"model": "simple_cnn", "input_shape": [1, 28, 28],
                                                 "device": "cpu", "batch_sizes": [4, 1], "warmup": 1,
                                                 "iterations": 3})
    assert result.status == "completed", result.error
    results = json.loads((folder / "results.json").read_text())
    assert [row["batch"] for row in results["rows"]] == [1, 4]  # sorted, each once
    assert all(row["mean_ms"] > 0 and row["throughput"] > 0 for row in results["rows"])
    assert results["parameters"] > 0 and results["device"] == "cpu" and results["run_id"] is None
    assert [e.payload["step"] for e in recorder.events("step")] == [1, 2]
    assert result.metrics["best_throughput"] == max(row["throughput"] for row in results["rows"])


def test_finished_sklearn_run(workspace) -> None:
    pytest.importorskip("sklearn")
    run_dir, _recorder, _result, _results = fit(workspace, make_spec(), "iris-run")
    relative = run_dir.relative_to(workspace.workspace).as_posix()
    folder, _rec, result = bench(workspace, {"run_dir": relative, "batch_sizes": [1, 16], "warmup": 1,
                                             "iterations": 3})
    assert result.status == "completed", result.error
    results = json.loads((folder / "results.json").read_text())
    assert results["framework"] == "sklearn" and results["run_id"] == "iris-run" and len(results["rows"]) == 2


def test_refuses_unknown_targets_and_bad_batches(workspace) -> None:
    _folder, _rec, result = bench(workspace, {"target": "os:system", "model": "simple_cnn"}, "evil")
    assert result.status == "failed" and "not a benchmark listed" in result.error
    _folder, _rec, result = bench(workspace, {"model": "simple_cnn", "batch_sizes": [0]}, "zero")
    assert result.status == "failed" and "batch sizes" in result.error
