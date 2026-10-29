"""Text classification with TF-IDF features and a linear model (logistic regression, linear SVM)
or naive Bayes — fast on a CPU, and every weight readable: the words that push a text towards
each class are part of the results.

A run is one fit, like the scikit-learn runner's, and leaves the same files, so the Training,
Evaluation and Compare pages show it unchanged: ``checkpoints/model.joblib`` (the vectoriser and
the model in one pipeline; a pickle the lab only loads when it wrote it), ``results.json``
(metrics, curves, the most telling words), ``test_confusion.json``, ``run_info.json`` and
``metrics.jsonl``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.common.optional import is_installed
from core.common.vocab import Task
from core.experiment_engine.runner import ExperimentRunner, RunCallbacks, RunContext, RunResult
from core.experiment_engine.spec import ExperimentSpec, SklearnTrainingSpec

from ..classical_ml.report import classification_report
from .text_data import load_splits, split_samples, text_adapter

__all__ = ["CLASSIFIERS", "MODEL_FILE", "TextClassifierRunner", "build_tfidf", "build_tfidf_linear_svm",
           "build_tfidf_logistic", "build_tfidf_naive_bayes", "telling_words"]

MODEL_FILE = "checkpoints/model.joblib"
TOP_WORDS = 15


def build_tfidf(*, task: str = "text_classification", seed: int = 42, classifier: str = "logistic",
                max_features: int = 50_000, ngram_max: int = 2, min_df: int = 2, sublinear_tf: bool = True,
                **params: Any) -> Any:
    """A pipeline: TF-IDF of words (1..ngram_max-grams) then the classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    if task != "text_classification":
        raise ValueError("TF-IDF models do text classification")
    if classifier == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(**{"max_iter": 2000, "C": 10.0, "random_state": seed, **params})
    elif classifier == "linear_svm":
        from sklearn.svm import LinearSVC

        model = LinearSVC(**{"C": 1.0, "random_state": seed, **params})
    elif classifier == "naive_bayes":
        from sklearn.naive_bayes import ComplementNB

        model = ComplementNB(**{"alpha": 0.3, **params})
    else:
        raise ValueError(f"unknown classifier {classifier!r} (logistic, linear_svm, naive_bayes)")
    vectorizer = TfidfVectorizer(max_features=int(max_features), ngram_range=(1, int(ngram_max)),
                                 min_df=int(min_df), sublinear_tf=bool(sublinear_tf), strip_accents="unicode")
    return Pipeline([("tfidf", vectorizer), ("model", model)])


def build_tfidf_logistic(**params: Any) -> Any:
    return build_tfidf(classifier="logistic", **params)


def build_tfidf_linear_svm(**params: Any) -> Any:
    return build_tfidf(classifier="linear_svm", **params)


def build_tfidf_naive_bayes(**params: Any) -> Any:
    return build_tfidf(classifier="naive_bayes", **params)


#: model card id → classifier.
CLASSIFIERS = {"tfidf_logistic": "logistic", "tfidf_linear_svm": "linear_svm",
               "tfidf_naive_bayes": "naive_bayes"}


def telling_words(pipeline: Any, classes: list[str], top: int = TOP_WORDS) -> dict[str, list[list[Any]]]:
    """Per class, the words with the largest weight towards it (linear models and naive Bayes)."""
    words = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_, dtype=float)
        if weights.shape[0] == 1:  # binary: one row, positive = class 1
            weights = np.vstack([-weights[0], weights[0]])
    elif hasattr(model, "feature_log_prob_"):
        logs = np.asarray(model.feature_log_prob_, dtype=float)
        weights = logs - logs.mean(axis=0, keepdims=True)
    else:
        return {}
    return {classes[c]: [[str(words[i]), float(weights[c, i])] for i in np.argsort(-weights[c])[:top]]
            for c in range(min(len(classes), weights.shape[0]))}


def _score(pipeline: Any, texts: list[str], y: np.ndarray, classes: list[str]) -> dict[str, Any]:
    predicted = np.asarray(pipeline.predict(texts), dtype=np.int64)
    proba = pipeline.predict_proba(texts) if hasattr(pipeline, "predict_proba") else None
    scores = pipeline.decision_function(texts) if proba is None and hasattr(pipeline,
                                                                            "decision_function") else None
    return classification_report(y, predicted, proba, classes, None if scores is None else np.asarray(scores))


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TextClassifierRunner(ExperimentRunner):
    id = "text_tfidf"
    title = "Text classification (TF-IDF)"
    tasks = frozenset({Task.TEXT_CLASSIFICATION})

    def supports(self, spec: ExperimentSpec) -> bool:
        return spec.task in self.tasks and spec.training.kind == "sklearn"

    def validate(self, spec: ExperimentSpec) -> list[str]:
        problems = super().validate(spec)
        if problems:
            return problems
        if not is_installed("scikit-learn"):
            problems.append("scikit-learn is needed in the experiment Python environment")
        try:
            paths, adapter, card = text_adapter(spec)
        except (LookupError, TypeError, ValueError, NotImplementedError) as exc:
            return [*problems, str(exc).strip("'\"")]
        if card.id not in CLASSIFIERS:
            problems.append(f"{card.name} is not a TF-IDF model; transformers train with training.kind "
                            "'torch'")
        if adapter.local_status(paths.datasets) != "present":
            problems.append(f"{adapter.card.name} is not downloaded yet: download it from the Dataset Hub or "
                            "the NLP / LLM lab")
        if spec.model.pretrained:
            problems.append(f"{card.name} has no pretrained weights; it is fitted from the data")
        if not problems:
            try:
                build_tfidf(**self._params(spec, card), seed=spec.seed)  # checks the options
            except (TypeError, ValueError) as exc:
                problems.append(f"model.params: {exc}")
        return problems

    @staticmethod
    def _params(spec: ExperimentSpec, card: Any) -> dict[str, Any]:
        params = dict(spec.model.params)
        if "classifier" in params:
            raise ValueError("the classifier is the model card; choose another model instead")
        return {"classifier": CLASSIFIERS.get(card.id, "logistic"), **params}

    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        import sklearn

        started = time.monotonic()
        paths, adapter, card = text_adapter(spec)
        training = spec.training
        assert isinstance(training, SklearnTrainingSpec)
        callbacks.on_epoch_start(1, 1)
        callbacks.on_step(1, 4, {})
        data = load_splits(spec, adapter, paths.datasets, training.test_fraction)
        callbacks.on_log(f"{adapter.card.name}: {len(data.train):,} training and {len(data.test):,} "
                         "test texts"
                         + (" (the dataset's own split)" if data.own_split else ""))
        pipeline = build_tfidf(**self._params(spec, card), seed=spec.seed)
        ctx.cancel.raise_if_cancelled()
        callbacks.on_step(2, 4, {})
        callbacks.on_log(f"Fitting {card.name}")
        pipeline.fit(data.train, data.y_train)
        ctx.cancel.raise_if_cancelled()
        callbacks.on_step(3, 4, {})
        train_report = _score(pipeline, data.train, data.y_train, data.classes)
        report = _score(pipeline, data.test, data.y_test, data.classes)
        metrics = {f"train_{k}": v for k, v in train_report["metrics"].items() if k in ("acc", "f1")}
        metrics.update({f"test_{k}": v for k, v in report.pop("metrics").items()})
        words = telling_words(pipeline, data.classes)
        importance = sorted(([f"{word} → {name}", weight, 0.0] for name, rows in words.items()
                             for word, weight in rows[:5]), key=lambda row: -row[1])
        results = {"task": spec.task.value, "metrics": metrics, **report, "words": words,
                   "importance": importance, "importance_kind": "words"}

        import joblib

        target = ctx.run_dir / MODEL_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target)
        callbacks.on_artifact(target, "model")
        info = {"model": card.name, "dataset": adapter.card.name, "classes": data.classes,
                "train_rows": len(data.train), "test_rows": len(data.test), "own_split": data.own_split,
                "vocabulary": len(pipeline.named_steps["tfidf"].vocabulary_), "sklearn": sklearn.__version__}
        _write_json(ctx.run_dir / "run_info.json", info)
        ctx.tracker.log_params(info)
        callbacks.on_artifact(_write_json(ctx.run_dir / "results.json", results), "results")
        callbacks.on_artifact(_write_json(ctx.run_dir / "test_confusion.json",
                                          {"classes": data.classes, "matrix": results["matrix"]}),
                              "confusion_matrix")
        callbacks.on_step(4, 4, {})
        callbacks.on_epoch_end(1, metrics)
        ctx.tracker.log_metrics(metrics, step=1, epoch=1)
        with (ctx.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": 1, **metrics}) + "\n")
        callbacks.on_log(f"Test accuracy {metrics['test_acc']:.4f} · macro F1 {metrics['test_f1']:.4f}")
        return RunResult(ctx.run_id, "completed", {**metrics, "epochs_completed": 1.0}, target,
                         time.monotonic() - started)

    def data_samples(self, spec: ExperimentSpec) -> dict[str, Any]:
        paths, adapter, _card = text_adapter(spec)
        training = spec.training
        assert isinstance(training, SklearnTrainingSpec)
        return split_samples(spec, load_splits(spec, adapter, paths.datasets, training.test_fraction))

    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        import joblib

        if split != "test":
            raise ValueError("text experiments with TF-IDF have no validation split; score them on test")
        paths, adapter, _card = text_adapter(spec)
        training = spec.training
        assert isinstance(training, SklearnTrainingSpec)
        data = load_splits(spec, adapter, paths.datasets, training.test_fraction)
        callbacks.on_log(f"Scoring {checkpoint.name} on {len(data.test):,} test texts")
        report = _score(joblib.load(checkpoint), data.test, data.y_test, data.classes)
        return {key: report[key] for key in ("metrics", "classes", "matrix")}
