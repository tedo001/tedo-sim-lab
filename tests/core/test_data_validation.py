"""Data checks before training: deliberately corrupted or overlapping splits are reported, and the
``fail`` rule stops the run (the acceptance test)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.common.cancel import CancelToken
from core.common.paths import AppPaths
from core.common.vocab import Task
from core.dataset_registry.validation import SampleSet, digest, validate
from core.experiment_engine.events import JsonLinesCallbacks
from core.experiment_engine.spec import (
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    SklearnTrainingSpec,
    TorchTrainingSpec,
    dump_spec,
)
from core.experiment_engine.worker import run_experiment


class Sink:
    def write(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


def statuses(report: dict) -> dict[str, str]:
    return {check["id"]: check["status"] for check in report["checks"]}


def test_clean_splits_pass():
    train = SampleSet([digest(f"a{i}") for i in range(40)], [i % 2 for i in range(40)], classes=("no", "yes"))
    test = SampleSet([digest(f"b{i}") for i in range(10)], [i % 2 for i in range(10)], classes=("no", "yes"))
    report = validate({"train": train, "test": test})
    found = statuses(report)
    assert report["errors"] == 0
    assert found["leakage:train:test"] == "ok" and found["label_conflicts"] == "ok"
    assert found["duplicates:train"] == "ok" and found["balance:train"] == "ok"


def test_overlap_conflicts_missing_and_imbalance_are_found():
    train = SampleSet(["x", "x", "y", *[f"r{i}" for i in range(30)]], [0, 0, 1, *([0] * 30)],
                      missing_labels=4, classes=("cat", "dog"))
    test = SampleSet(["y", "z", "x"], [0, 1, 0], classes=("cat", "dog"))  # y is "dog" in train, "cat" here
    report = validate({"train": train, "test": test})
    found = statuses(report)
    assert found["duplicates:train"] == "warn"
    assert found["leakage:train:test"] == "error"
    assert found["label_conflicts"] == "error" and found["missing_labels"] == "error"
    assert found["balance:train"] == "warn"
    leak = next(c for c in report["checks"] if c["id"] == "leakage:train:test")
    assert leak["count"] == 2 and "also appear in train" in leak["detail"]
    assert report["errors"] == 3


@pytest.fixture
def paths(tmp_path, monkeypatch):
    found = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(found.workspace))
    return found


def run(paths: AppPaths, spec: ExperimentSpec, name: str):
    run_dir = paths.experiments / name / "run"
    dump_spec(spec, run_dir / "experiment.yaml")
    result = run_experiment(run_dir, JsonLinesCallbacks(Sink(), "run"), CancelToken())  # type: ignore[arg-type]
    report_file = run_dir / "data_validation.json"
    return result, json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else None


def corrupted_csv(paths: AppPaths, tmp_path) -> str:
    pytest.importorskip("sklearn")
    from labs.classical_ml.datasets import import_csv

    rng = np.random.default_rng(0)
    rows = [f"{a:.3f},{b:.3f},{'red' if a > 0 else 'blue'}" for a, b in rng.normal(size=(80, 2))]
    rows += rows[:30]                     # thirty rows copied: they land in both splits
    rows.append(rows[0].rsplit(",", 1)[0] + ",green")  # the same row with another label
    rows += ["0.5,0.5,", "0.1,0.2,"]      # no label
    source = tmp_path / "corrupted.csv"
    source.write_text("a,b,colour\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return import_csv(source, paths.datasets, name="Corrupted", target="colour",
                      task="tabular_classification")


def tabular(card_id: str, rule: str) -> ExperimentSpec:
    return ExperimentSpec(name=f"check-{rule}", task=Task.TABULAR_CLASSIFICATION,
                          data=DataSpec(dataset=card_id, validation=rule),
                          model=ModelSpec(model="logistic_regression"), training=SklearnTrainingSpec())


def test_corrupted_table_warns_and_the_fail_rule_stops_the_run(paths, tmp_path):
    card_id = corrupted_csv(paths, tmp_path)
    result, report = run(paths, tabular(card_id, "warn"), "warn")
    assert result.status == "completed"  # "warn" records and carries on
    found = statuses(report)
    assert found["leakage:train:test"] == "error" and found["label_conflicts"] == "error"
    assert found["missing_labels"] == "error" and found["licence"] == "warn"  # an imported CSV has no licence
    assert report["rule"] == "warn"
    result, report = run(paths, tabular(card_id, "fail"), "fail")
    assert result.status == "failed" and "data checks found" in result.error
    assert "rule is 'fail'" in result.error
    assert report["errors"] >= 3
    result, report = run(paths, tabular(card_id, "off"), "off")
    assert result.status == "completed" and report is None


def test_test_images_copied_into_training_are_leakage(paths):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tests.fixtures.fake_vision import _idx, make_images, write_fake_mnist

    raw = write_fake_mnist(paths.datasets, train=120, test=40)
    train, train_labels = make_images(120, 1)
    test, test_labels = make_images(40, 2)
    train[:15], train_labels[:15] = test[:15], test_labels[:15]  # a careless merge of the test set
    _idx(raw / "train-images-idx3-ubyte", train, 3)
    _idx(raw / "train-labels-idx1-ubyte", train_labels, 1)
    spec = ExperimentSpec(name="leaky-mnist", task=Task.IMAGE_CLASSIFICATION,
                          data=DataSpec(dataset="mnist", validation="fail"),
                          model=ModelSpec(model="simple_cnn"),
                          training=TorchTrainingSpec(epochs=1, batch_size=16))
    result, report = run(paths, spec, "leaky")
    assert result.status == "failed" and "also appear in train" in result.error
    leaks = {c["id"]: c for c in report["checks"] if c["id"].startswith("leakage")}
    assert leaks["leakage:train:test"]["count"] == 15 or leaks["leakage:val:test"]["count"] > 0
    assert sum(c["count"] for c in leaks.values() if c["id"].endswith(":test")) == 15
