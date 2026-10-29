"""Reproducibility: every finished run leaves a report and a bundle, and the bundle reproduces the
run's numbers in a clean workspace (the acceptance test)."""

from __future__ import annotations

import json
import zipfile

import pytest

pytest.importorskip("sklearn")

from core.common.cancel import CancelToken  # noqa: E402
from core.common.paths import (
    CODE_ROOT,  # noqa: E402
    AppPaths,  # noqa: E402
)
from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.events import JsonLinesCallbacks  # noqa: E402
from core.experiment_engine.repro import bundle, find_run, wilson_interval  # noqa: E402
from core.experiment_engine.reproduce import compare_scores, main, reproduce  # noqa: E402
from core.experiment_engine.snapshot import compare, environment  # noqa: E402
from core.experiment_engine.spec import (  # noqa: E402
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    ResearchSpec,
    SklearnTrainingSpec,
    dump_spec,
    spec_hash,
)
from core.experiment_engine.worker import run_experiment  # noqa: E402


class Sink:
    def write(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


def iris_spec(**research) -> ExperimentSpec:
    return ExperimentSpec(name="iris-logreg", task=Task.TABULAR_CLASSIFICATION,
                          data=DataSpec(dataset="iris"), model=ModelSpec(model="logistic_regression"),
                          training=SklearnTrainingSpec(), research=ResearchSpec(**research))


@pytest.fixture
def first(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "a").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def train(paths: AppPaths, spec: ExperimentSpec, run_id: str = "run1"):
    run_dir = paths.experiments / f"{spec.name}-exp" / run_id
    dump_spec(spec, run_dir / "experiment.yaml")
    result = run_experiment(run_dir, JsonLinesCallbacks(Sink(), run_id), CancelToken())  # type: ignore[arg-type]
    assert result.status == "completed", result.error
    return run_dir, result


def test_every_run_writes_a_report_with_all_its_parts(first):
    run_dir, result = train(first, iris_spec(question="Is a linear model enough for Iris?",
                                             hypothesis="Logistic regression reaches 90 % test accuracy.",
                                             limitations="Iris is tiny."))
    folder = run_dir / "reproduction"
    for name in ("report.md", "report.json", "REPRODUCE.md", "experiment.yaml", "snapshot.json",
                 "environment.lock.txt", "result.json"):
        assert (folder / name).is_file(), name
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["question"].startswith("Is a linear model")
    assert report["hypothesis"].startswith("Logistic regression")
    assert report["provenance"]["dataset"] == "iris" and report["provenance"]["licence"]
    assert report["environment"]["python"] and report["environment"]["lock"]["count"] > 10
    assert report["results"]["test_acc"] == pytest.approx(result.metrics["test_acc"])
    assert report["intervals"]["test_acc"]["low"] <= report["results"]["test_acc"] <= 1
    observed = " ".join(report["limitations"]["observed"])
    assert "One seed" in observed and "test samples" in observed
    text = (folder / "report.md").read_text(encoding="utf-8")
    for heading in ("## Question", "## Hypothesis", "## Data provenance", "## Environment", "## Results",
                    "## Baseline", "## Limitations", "## Configuration", "## Reproduce"):
        assert heading in text
    assert "python -m core.experiment_engine.reproduce" in text and "Iris is tiny." in text


def test_research_fields_never_change_the_spec_hash():
    assert spec_hash(iris_spec()) == spec_hash(iris_spec(question="why", hypothesis="because"))


def test_a_baseline_run_is_compared(first):
    train(first, iris_spec(), "base")
    run_dir, _ = train(first, iris_spec(baseline="base"), "next")
    report = json.loads((run_dir / "reproduction" / "report.json").read_text(encoding="utf-8"))
    assert report["baseline"]["run_id"] == "base"
    assert report["baseline"]["delta"]["test_acc"] == pytest.approx(0.0)
    assert find_run(first.experiments, "base") is not None


def test_bundle_reproduces_in_a_clean_workspace(first, tmp_path, monkeypatch):
    run_dir, result = train(first, iris_spec())
    archive = bundle(run_dir, tmp_path / "out" / "iris.zip")
    assert "experiment.yaml" in {name.split("/")[-1] for name in zipfile.ZipFile(archive).namelist()}
    clean = tmp_path / "b"  # another person's empty workspace
    verdict = reproduce(archive, clean)
    assert verdict["status"] == "completed", verdict.get("error")
    assert verdict["reproduced"], verdict["scores"]
    scores = {row["metric"]: row for row in verdict["scores"]}
    assert scores["test_acc"]["reproduced"] == pytest.approx(result.metrics["test_acc"])
    assert str(clean) in verdict["run_dir"]
    assert isinstance(verdict["environment_differences"], list)


def test_reproduce_cli_exit_codes(first, tmp_path, capsys):
    run_dir, _ = train(first, iris_spec())
    assert main([str(run_dir), "--workspace", str(tmp_path / "c")]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reproduced"] is True
    recorded = json.loads((run_dir / "reproduction" / "result.json").read_text(encoding="utf-8"))
    recorded["metrics"]["test_acc"] = -1.0  # a number the run never produced
    (run_dir / "reproduction" / "result.json").write_text(json.dumps(recorded), encoding="utf-8")
    assert main([str(run_dir), "--workspace", str(tmp_path / "d")]) == 3


def test_compare_scores_skips_timings_and_flags_differences():
    rows = compare_scores({"test_acc": 0.9, "epoch_time_s": 3.0, "val_loss": 0.2},
                          {"test_acc": 0.9, "epoch_time_s": 9.0, "val_loss": 0.3}, 1e-9)
    assert {row["metric"]: row["match"] for row in rows} == {"test_acc": True, "val_loss": False}


def test_wilson_interval():
    low, high = wilson_interval(90, 100)
    assert 0.82 < low < 0.9 < high < 0.95
    assert wilson_interval(0, 0) is None


def test_environment_records_the_package_footprint():
    env = environment(CODE_ROOT)
    assert env["python"] and env["os"]["system"] and "cpu" in env["hardware"]
    assert "cuda_toolkit" in env["torch_build"] or not env["packages"].get("torch")
    changed = {**env, "torch_build": {"cuda_toolkit": "12.1"}, "lock": {"sha256": "x", "count": 1}}
    notes = " ".join(compare({**env, "deterministic": True}, changed))
    assert "CUDA toolkit" in notes and "installed packages differ" in notes
