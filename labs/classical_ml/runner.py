"""The scikit-learn / XGBoost runner: tabular classification and regression, clustering and
dimensionality reduction on the built-in tables or an imported CSV file.

Supervised tasks hold out ``training.test_fraction`` of the rows (stratified for classes), fit
the pipeline (imputation, encoding, optional scaling and feature selection, the model) on the
rest, optionally inside a grid or random search, cross-validate when asked, and score the test
rows once. Unsupervised tasks fit every row. Files in the run folder:
``checkpoints/model.joblib`` (the fitted pipeline), ``results.json`` (metrics, curves,
importance, 2-D points), ``test_confusion.json`` (classification), ``run_info.json`` and
``metrics.jsonl``. ``model.joblib`` is a pickle: the lab only loads the ones it wrote itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.catalog import load_cards
from core.common.optional import is_installed
from core.common.paths import AppPaths
from core.common.vocab import TABULAR_TASKS, Task
from core.dataset_registry.adapter import TabularAdapter
from core.experiment_engine.runner import ExperimentRunner, RunCallbacks, RunContext, RunResult
from core.experiment_engine.spec import ExperimentSpec, SklearnTrainingSpec

from .explain import shap_summary
from .pipeline import build_pipeline, pipeline_problems, wrap_search
from .report import classification_report, cluster_report, importance, projection_report, regression_report
from .tables import SUPERVISED, Table, load_table, split_table

__all__ = ["MODEL_FILE", "SCORING", "SklearnRunner"]

MODEL_FILE = "checkpoints/model.joblib"
#: The score a search or cross-validation optimises by default, per task.
SCORING = {Task.TABULAR_CLASSIFICATION: "accuracy", Task.TABULAR_REGRESSION: "r2"}


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _score(pipeline: Any, task: Task, x: Any, y: np.ndarray, classes: list[str]) -> dict[str, Any]:
    predicted = np.asarray(pipeline.predict(x))
    if task == Task.TABULAR_CLASSIFICATION:
        proba = pipeline.predict_proba(x) if hasattr(pipeline, "predict_proba") else None
        scores = pipeline.decision_function(x) if proba is None and hasattr(pipeline, "decision_function") \
            else None
        return classification_report(y, predicted.astype(np.int64), proba, classes, np.asarray(scores)
                                     if scores is not None else None)
    return regression_report(y, predicted.astype(float))


class SklearnRunner(ExperimentRunner):
    id = "sklearn"
    title = "scikit-learn and XGBoost"
    tasks = TABULAR_TASKS

    # Checks --------------------------------------------------------------------
    def _parts(self, spec: ExperimentSpec) -> tuple[AppPaths, TabularAdapter, Any, Any]:
        paths = AppPaths.resolve()
        datasets, models = load_cards(paths)
        if spec.data.dataset not in datasets:
            raise LookupError(f"no dataset card {spec.data.dataset!r}")
        adapter = datasets.adapter(spec.data.dataset)
        if not isinstance(adapter, TabularAdapter):
            raise TypeError(f"{adapter.card.name} is not a tabular dataset")
        if spec.model.model not in models:
            raise LookupError(f"no model card {spec.model.model!r}")
        card = models.get(spec.model.model)
        if spec.task not in card.tasks:
            raise ValueError(f"{card.name} does not do {spec.task.value.replace('_', ' ')}")
        if spec.model.pretrained:
            raise ValueError(f"{card.name} has no pretrained weights; it is fitted from the data")
        return paths, adapter, card, models.builder(spec.model.model)

    def validate(self, spec: ExperimentSpec) -> list[str]:
        problems = super().validate(spec)
        if problems:
            return problems
        if not is_installed("scikit-learn"):
            problems.append("scikit-learn is needed in the experiment Python environment")
        if spec.model.model == "xgboost" and not is_installed("xgboost"):
            problems.append("XGBoost is not installed in the experiment Python environment")
        try:
            paths, adapter, _card, builder = self._parts(spec)
        except (LookupError, TypeError, ValueError, NotImplementedError) as exc:
            return [*problems, str(exc).strip("'\"")]
        if adapter.local_status(paths.datasets) != "present":
            problems.append(f"{adapter.card.name} is not available: "
                            + ("install scikit-learn" if "sklearn" in adapter.card.tags
                               else "import the CSV file again"))
        problems += pipeline_problems(spec)
        training = spec.training
        assert isinstance(training, SklearnTrainingSpec)
        if spec.task not in SUPERVISED:
            if training.search is not None:
                problems.append("search needs a score to optimise; it is for classification and regression")
            if training.cv_folds:
                problems.append("cross-validation is for classification and regression")
        if not problems and is_installed("scikit-learn"):
            try:
                builder(task=spec.task.value, seed=spec.seed, **spec.model.params)
            except (TypeError, ValueError) as exc:
                problems.append(f"model.params: {exc}")
        return problems

    # Run -----------------------------------------------------------------------
    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        import sklearn

        started = time.monotonic()
        paths, adapter, card, builder = self._parts(spec)
        training = spec.training
        assert isinstance(training, SklearnTrainingSpec)
        supervised = spec.task in SUPERVISED
        stages = 4 + bool(training.cv_folds and training.search is None)
        stage = 0

        def advance(text: str) -> None:
            nonlocal stage
            ctx.cancel.raise_if_cancelled()
            stage += 1
            callbacks.on_step(stage, stages, {})
            callbacks.on_log(text)

        callbacks.on_epoch_start(1, 1)
        table = load_table(spec, adapter, paths.datasets)
        model = builder(task=spec.task.value, seed=spec.seed, **spec.model.params)
        pipeline = build_pipeline(spec, model, table.features.columns, supervised=supervised)
        info: dict[str, Any] = {"model": card.name, "dataset": adapter.card.name, "rows": len(table.features),
                                "features": list(map(str, table.features.columns)),
                                "target": table.target_name, "classes": table.classes,
                                "sklearn": sklearn.__version__}
        if spec.model.model == "xgboost":
            import xgboost

            info["xgboost"] = xgboost.__version__
        advance(f"{adapter.card.name}: {len(table.features):,} rows × {table.features.shape[1]} columns"
                + (f", target {table.target_name!r}" if table.target_name else ""))

        if supervised:
            results, metrics = self._supervised(spec, training, pipeline, table, info, advance, callbacks)
        else:
            results, metrics = self._unsupervised(spec, pipeline, table, advance)
        pipeline = results.pop("_fitted")
        ctx.cancel.raise_if_cancelled()

        import joblib

        target = ctx.run_dir / MODEL_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target)
        callbacks.on_artifact(target, "model")
        info["encoded_features"] = int(np.asarray(pipeline[:-1].transform(table.features.head(1))).shape[1])
        _write_json(ctx.run_dir / "run_info.json", info)
        ctx.tracker.log_params(info)
        results.update({"task": spec.task.value, "metrics": metrics})
        callbacks.on_artifact(_write_json(ctx.run_dir / "results.json", results), "results")
        if spec.task == Task.TABULAR_CLASSIFICATION:
            callbacks.on_artifact(_write_json(ctx.run_dir / "test_confusion.json",
                                              {"classes": results["classes"], "matrix": results["matrix"]}),
                                  "confusion_matrix")
        callbacks.on_epoch_end(1, metrics)
        ctx.tracker.log_metrics(metrics, step=1, epoch=1)
        with (ctx.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": 1, **metrics}) + "\n")
        advance("Done · " + " · ".join(f"{key} {value:.4f}" for key, value in list(metrics.items())[:4]))
        return RunResult(ctx.run_id, "completed", {**metrics, "epochs_completed": 1.0}, target,
                         time.monotonic() - started)

    def _supervised(self, spec: ExperimentSpec, training: SklearnTrainingSpec, pipeline: Any, table: Table,
                    info: dict[str, Any], advance: Any, callbacks: RunCallbacks) -> tuple[dict, dict]:
        from sklearn.model_selection import cross_val_score

        scoring = SCORING[spec.task]
        x_train, x_test, y_train, y_test = split_table(spec, table)
        info.update(train_rows=len(x_train), test_rows=len(x_test))
        results: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        if training.cv_folds and training.search is None:
            advance(f"Cross-validating on {training.cv_folds} folds of {len(x_train):,} training rows")
            scores = cross_val_score(pipeline, x_train, y_train, cv=training.cv_folds, scoring=scoring)
            results["cv"] = {"scoring": scoring, "folds": [float(s) for s in scores]}
            metrics.update(cv_mean=float(scores.mean()), cv_std=float(scores.std()))
        if training.search is not None:
            search = wrap_search(pipeline, training, spec.seed, scoring)
            advance(f"{training.search.method.capitalize()} search over {', '.join(training.search.space)} "
                    f"with {search.cv}-fold cross-validation")
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            order = np.argsort(search.cv_results_["rank_test_score"])[:10]

            def plain(params: dict[str, Any]) -> dict[str, Any]:
                return {key.removeprefix("model__"): value for key, value in params.items()}

            results["search"] = {
                "scoring": training.search.scoring or scoring, "best_score": float(search.best_score_),
                "best_params": plain(search.best_params_),
                "top": [{"params": plain(search.cv_results_["params"][i]),
                         "mean": float(search.cv_results_["mean_test_score"][i]),
                         "std": float(search.cv_results_["std_test_score"][i])} for i in order]}
            metrics["cv_mean"] = float(search.best_score_)
            callbacks.on_log(f"Best {results['search']['scoring']} {search.best_score_:.4f} with "
                             f"{results['search']['best_params']}")
        else:
            advance(f"Fitting {spec.model.model} on {len(x_train):,} rows")
            fitted = pipeline.fit(x_train, y_train)
        advance(f"Scoring {len(x_test):,} held-out test rows")
        train_report = _score(fitted, spec.task, x_train, y_train, table.classes)
        report = _score(fitted, spec.task, x_test, y_test, table.classes)
        metrics.update({f"train_{key}": value for key, value in train_report["metrics"].items()
                        if key in ("acc", "f1", "r2", "rmse")})
        metrics.update({f"test_{key}": value for key, value in report.pop("metrics").items()})
        results.update(report)
        results["importance"] = importance(fitted, x_test, y_test, scoring, spec.seed)
        try:
            explained = shap_summary(fitted, x_test, spec.model.model, seed=spec.seed)
        except Exception as exc:  # SHAP is optional; its failure never fails the run
            explained = None
            callbacks.on_log(f"SHAP values skipped: {type(exc).__name__}: {exc}")
        if explained is not None:
            results["shap"] = explained
        results["_fitted"] = fitted
        return results, metrics

    def _unsupervised(self, spec: ExperimentSpec, pipeline: Any, table: Table,
                      advance: Any) -> tuple[dict, dict]:
        advance(f"Fitting {spec.model.model} on all {len(table.features):,} rows")
        pipeline.fit(table.features)
        encoded = np.asarray(pipeline[:-1].transform(table.features), dtype=float)
        model = pipeline[-1]
        advance("Measuring the result")
        if spec.task == Task.CLUSTERING:
            labels = np.asarray(model.labels_ if hasattr(model, "labels_") else model.predict(encoded))
            report = cluster_report(model, encoded, labels.astype(np.int64), table.truth, spec.seed)
            report["groups"] = [f"cluster {i}" for i in range(len(report["sizes"]))]
        else:
            report = projection_report(model, np.asarray(model.transform(encoded)), table.truth, spec.seed)
            report["groups"] = table.truth_names or []
        report["_fitted"] = pipeline
        return report, report.pop("metrics")

    # Re-scoring ----------------------------------------------------------------
    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        import joblib

        if spec.task not in SUPERVISED:
            raise ValueError("clustering and projections have no held-out rows to score")
        if split != "test":
            raise ValueError("scikit-learn experiments have no validation split; their cross-validation "
                             "scores are in results.json")
        paths, adapter, _card, _builder = self._parts(spec)
        table = load_table(spec, adapter, paths.datasets)
        _x_train, x_test, _y_train, y_test = split_table(spec, table)
        callbacks.on_log(f"Scoring {checkpoint.name} on {len(x_test):,} test rows")
        pipeline = joblib.load(checkpoint)
        report = _score(pipeline, spec.task, x_test, y_test, table.classes)
        return {key: report[key] for key in ("metrics", "classes", "matrix") if key in report}
