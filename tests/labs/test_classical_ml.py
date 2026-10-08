"""The scikit-learn / XGBoost runner end to end: built-in tables, an imported CSV file, every
task, search and cross-validation, re-scoring, and the checks that stop a bad spec."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from core.catalog import load_cards  # noqa: E402
from core.common.cancel import CancelToken  # noqa: E402
from core.common.optional import is_installed  # noqa: E402
from core.common.paths import AppPaths  # noqa: E402
from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.events import JsonLinesCallbacks, parse_line  # noqa: E402
from core.experiment_engine.spec import (  # noqa: E402
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    SearchSpec,
    SklearnTrainingSpec,
    TransformSpec,
    dump_spec,
)
from core.experiment_engine.worker import run_evaluation, run_experiment  # noqa: E402
from labs.classical_ml.datasets import import_csv, preview_csv  # noqa: E402
from labs.classical_ml.models import BUILDERS  # noqa: E402
from labs.classical_ml.presets import CLASSICAL_PRESETS  # noqa: E402
from labs.classical_ml.runner import SklearnRunner  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def make_spec(dataset: str = "iris", model: str = "logistic_regression",
              task: Task = Task.TABULAR_CLASSIFICATION, *, scale: bool = True, params: dict | None = None,
              **training) -> ExperimentSpec:
    return ExperimentSpec(name=f"{dataset}-{model}", task=task,
                          data=DataSpec(dataset=dataset, preprocessing=(TransformSpec(name="standardize"),)
                                        if scale else ()),
                          model=ModelSpec(model=model, params=params or {}),
                          training=SklearnTrainingSpec(**training))


class Recorder(JsonLinesCallbacks):
    def __init__(self) -> None:
        self.lines: list[str] = []
        super().__init__(self, "run")  # type: ignore[arg-type]

    def write(self, text: str) -> None:
        self.lines += [line for line in text.splitlines() if line]

    def flush(self) -> None:
        pass

    def events(self, kind: str) -> list:
        return [event for event in map(parse_line, self.lines) if event.type == kind]


def fit(paths: AppPaths, spec: ExperimentSpec, name: str = "run"):
    run_dir = paths.experiments / name
    dump_spec(spec, run_dir / "experiment.yaml")
    recorder = Recorder()
    result = run_experiment(run_dir, recorder, CancelToken())
    assert result.status == "completed", result.error
    return run_dir, recorder, result, json.loads((run_dir / "results.json").read_text())


def test_builders_build_for_their_tasks_and_refuse_others() -> None:
    for name, builder in BUILDERS.items():
        if name == "xgboost" and not is_installed("xgboost"):
            continue
        tasks = {"pca": ["dimensionality_reduction"], "kmeans": ["clustering"],
                 "linear_regression": ["tabular_regression"],
                 "logistic_regression": ["tabular_classification"],
                 "naive_bayes": ["tabular_classification"]}.get(name, ["tabular_classification",
                                                                      "tabular_regression"])
        for task in tasks:
            assert hasattr(builder(task=task), "fit")
    with pytest.raises(ValueError, match="does not do"):
        BUILDERS["naive_bayes"](task="tabular_regression")
    with pytest.raises(ValueError, match="clustering"):
        BUILDERS["kmeans"](task="tabular_classification")
    assert BUILDERS["random_forest"](task="tabular_classification").random_state == 42


def test_validate_explains_problems(workspace) -> None:
    runner = SklearnRunner()
    assert runner.validate(make_spec()) == []
    assert "does not do tabular regression" in runner.validate(make_spec(model="naive_bayes",
                                                                         task=Task.TABULAR_REGRESSION))[0]
    assert "is not a tabular dataset" in runner.validate(make_spec(dataset="mnist"))[0]
    odd = make_spec().model_copy(update={"data": DataSpec(dataset="iris", preprocessing=(
        TransformSpec(name="standardize"), TransformSpec(name="minmax"), TransformSpec(name="whiten")))})
    problems = runner.validate(odd)
    assert any("'whiten'" in p for p in problems) and any("one scaling" in p for p in problems)
    clustering = make_spec(model="kmeans", task=Task.CLUSTERING, cv_folds=3,
                           search=SearchSpec(space={"n_clusters": (2, 3)}))
    assert len(runner.validate(clustering)) == 2
    assert "model.params" in runner.validate(make_spec(params={"no_such_option": 1}))[0]


def test_classification_end_to_end_with_cross_validation(workspace) -> None:
    run_dir, recorder, result, results = fit(workspace, make_spec(cv_folds=5))
    assert [event.payload["epoch"] for event in recorder.events("epoch_end")] == [1]
    assert recorder.events("step")[-1].payload["step"] == recorder.events("step")[-1].payload["total"]
    assert result.metrics["test_acc"] > 0.85 and 0 < result.metrics["cv_mean"] <= 1
    assert len(results["cv"]["folds"]) == 5
    assert results["classes"] == ["setosa", "versicolor", "virginica"]
    assert np.asarray(results["matrix"]).sum() == 30  # 20% of 150 rows
    assert {curve["class"] for curve in results["curves"]} == set(results["classes"])
    assert all(len(curve["roc"]) <= 200 for curve in results["curves"])
    assert results["importance"][0][0].startswith("petal")
    confusion = json.loads((run_dir / "test_confusion.json").read_text())
    assert confusion["matrix"] == results["matrix"]
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["train_rows"] == 120 and info["encoded_features"] == 4
    model = result.best_checkpoint
    assert (run_dir / "checkpoints" / "model.joblib").is_file() and model.name == "model.joblib"
    # Re-scoring the saved model on the same split gives the same numbers.
    again = run_evaluation(run_dir, Recorder(), CancelToken(), checkpoint=model, split="test")
    assert again.metrics["test_acc"] == pytest.approx(result.metrics["test_acc"])
    refused = run_evaluation(run_dir, Recorder(), CancelToken(), checkpoint=model, split="val")
    assert refused.status == "failed" and "no validation split" in refused.error


def test_grid_search_picks_parameters_and_svm_scores_rank(workspace) -> None:
    spec = make_spec("wine", "svm", search=SearchSpec(space={"C": (0.1, 10.0)}), cv_folds=3)
    _run_dir, _recorder, result, results = fit(workspace, spec)
    assert results["search"]["best_params"]["C"] in (0.1, 10.0)
    assert len(results["search"]["top"]) == 2 and result.metrics["cv_mean"] == results["search"]["best_score"]
    assert results["curves"] and "test_log_loss" not in result.metrics  # decision scores, no probabilities


def test_regression_end_to_end(workspace) -> None:
    spec = make_spec("diabetes", "linear_regression", Task.TABULAR_REGRESSION, scale=False)
    _run_dir, _recorder, result, results = fit(workspace, spec)
    assert result.metrics["test_r2"] > 0.3 and result.metrics["test_rmse"] > 0
    assert len(results["predicted_vs_actual"]) == 89
    assert "classes" not in results


@pytest.mark.skipif(not is_installed("xgboost"), reason="XGBoost is not installed")
def test_xgboost_on_breast_cancer(workspace) -> None:
    spec = make_spec("breast_cancer", "xgboost", scale=False, params={"n_estimators": 30})
    _run_dir, _recorder, result, results = fit(workspace, spec)
    assert result.metrics["test_acc"] > 0.9 and result.metrics["test_roc_auc"] > 0.95
    if is_installed("shap"):
        assert results["shap"][0][1] >= results["shap"][-1][1] > -1


def test_clustering_and_projection(workspace) -> None:
    _d, _r, result, results = fit(workspace, make_spec(model="kmeans", task=Task.CLUSTERING), "kmeans")
    assert result.metrics["clusters"] == 3 and sum(results["sizes"]) == 150
    assert 0 < result.metrics["adjusted_rand"] <= 1 and len(results["points"]) == 150
    _d, _r, result, results = fit(workspace, make_spec("digits", "pca", Task.DIMENSIONALITY_REDUCTION), "pca")
    assert len(results["explained_variance_ratio"]) == 2 and 0 < result.metrics["explained_variance"] < 1
    assert {point[2] for point in results["points"]} == set(range(10))


def test_csv_import_and_training(workspace, tmp_path) -> None:
    rng = np.random.default_rng(0)
    rows = ["size,colour,rooms,price"]
    for i in range(120):
        size, rooms = rng.uniform(40, 200), int(rng.integers(1, 6))
        colour = ("red", "blue", "green")[i % 3]
        price = 3 * size + 20 * rooms + (15 if colour == "red" else 0) + rng.normal(0, 5)
        rows.append(f"{'' if i % 17 == 0 else round(size, 1)},{colour},{rooms},{price:.1f}")
    source = tmp_path / "houses.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    preview = preview_csv(source)
    assert preview.kinds == {"size": "number", "colour": "text", "rooms": "number", "price": "number"}
    assert preview.guess_task("price") == "tabular_regression" and preview.guess_task("colour") == \
        "tabular_classification"
    card_id = import_csv(source, workspace.datasets, name="Houses", target="price", task="tabular_regression")
    assert card_id == "csv-houses"
    assert import_csv(source, workspace.datasets, name="Houses", target="colour",
                      task="tabular_classification") == "csv-houses-2"
    datasets, _models = load_cards(workspace)
    assert datasets.get(card_id).license.category == "unspecified"
    assert datasets.status(card_id) == "ready"
    spec = make_spec(card_id, "random_forest", Task.TABULAR_REGRESSION, params={"n_estimators": 50})
    _d, _r, result, results = fit(workspace, spec, "houses")
    assert result.metrics["test_r2"] > 0.8  # missing sizes imputed, colour one-hot encoded
    assert {row[0] for row in results["importance"]} == {"size", "colour", "rooms"}
    # The training spec can pick another target and a subset of the features.
    spec = make_spec(card_id, "decision_tree", target="colour", features=("size", "price"))
    _d, _r, result, results = fit(workspace, spec, "colours")
    assert results["classes"] == ["blue", "green", "red"]


def test_presets_are_valid_and_runnable(workspace) -> None:
    runner = SklearnRunner()
    for preset in CLASSICAL_PRESETS:
        spec = preset.make()
        if spec.model.model == "xgboost" and not is_installed("xgboost"):
            continue
        assert runner.validate(spec) == [], preset.id
