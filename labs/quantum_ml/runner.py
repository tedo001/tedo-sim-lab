"""The quantum-classifier runner: a variational circuit (``vqc``) trained epoch by epoch, or a
quantum-kernel SVM (``quantum_kernel_svm``) fitted once, on any tabular dataset with a target.

Files in the run folder: ``checkpoints/last.npz`` (every epoch: parameters, optimiser state,
epoch; the resume point) and ``best.npz`` (best validation accuracy), ``metrics.jsonl``,
``run_info.json``, ``results.json`` (test scores, the circuit, the decision boundary for
two-dimensional data, the kernel matrix) and ``test_confusion.json``. Batches are shuffled with
a generator seeded by the experiment seed and the epoch, so a resumed run continues exactly as
an uninterrupted one would.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.catalog import load_cards
from core.common.cancel import Cancelled
from core.common.paths import AppPaths
from core.common.vocab import Task
from core.dataset_registry.adapter import TabularAdapter
from core.experiment_engine.runner import ExperimentRunner, RunCallbacks, RunContext, RunResult
from core.experiment_engine.spec import ExperimentSpec, QuantumTrainingSpec

from .data import Prepared, prepare
from .kernel import fidelity_kernel, kernel_alignment
from .sim import ENCODINGS
from .vqc import Adam, VariationalClassifier

__all__ = ["MODELS", "QuantumRunner", "build_kernel_svm", "build_vqc"]

MODELS = ("vqc", "quantum_kernel_svm")
DEFAULTS = {"vqc": {"qubits": 4, "layers": 2, "encoding": "angle"},
            "quantum_kernel_svm": {"qubits": 4, "encoding": "zz", "C": 1.0}}
_GRID = 48


def build_vqc(*, classes: int = 2, seed: int = 42, **params: Any) -> VariationalClassifier:
    """The model card's builder: an untrained variational classifier with ``model.params``."""
    params = {**DEFAULTS["vqc"], **params}
    return VariationalClassifier(int(params["qubits"]), classes, layers=int(params["layers"]),
                                 encoding=params["encoding"], seed=seed)


def build_kernel_svm(*, seed: int = 42, **params: Any) -> Any:
    """The model card's builder: an SVM for a precomputed quantum kernel."""
    from sklearn.svm import SVC

    params = {**DEFAULTS["quantum_kernel_svm"], **params}
    return SVC(kernel="precomputed", C=float(params["C"]), probability=True, random_state=seed)


def _json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


class QuantumRunner(ExperimentRunner):
    id = "quantum"
    title = "Quantum classifiers"
    tasks = frozenset({Task.QUANTUM_CLASSIFICATION})

    def _params(self, spec: ExperimentSpec) -> dict[str, Any]:
        unknown = set(spec.model.params) - set(DEFAULTS.get(spec.model.model, {}))
        if unknown:
            raise ValueError(f"model.params: unknown setting(s) {', '.join(sorted(unknown))}")
        params = {**DEFAULTS[spec.model.model], **spec.model.params}
        if params["encoding"] not in ENCODINGS:
            raise ValueError(f"model.params.encoding: {params['encoding']!r} (known: {', '.join(ENCODINGS)})")
        if not 2 <= int(params["qubits"]) <= 10:
            raise ValueError("model.params.qubits: between 2 and 10")
        return params

    def _parts(self, spec: ExperimentSpec) -> tuple[AppPaths, TabularAdapter]:
        paths = AppPaths.resolve()
        datasets, _models = load_cards(paths)
        if spec.data.dataset not in datasets:
            raise LookupError(f"no dataset card {spec.data.dataset!r}")
        adapter = datasets.adapter(spec.data.dataset)
        if not isinstance(adapter, TabularAdapter):
            raise TypeError(f"{adapter.card.name} is not a table; quantum classifiers take tabular data")
        if spec.model.model not in MODELS:
            raise LookupError(f"{spec.model.model!r} "
                              f"is not a quantum classifier (known: {', '.join(MODELS)})")
        return paths, adapter

    def validate(self, spec: ExperimentSpec) -> list[str]:
        problems = super().validate(spec)
        if problems:
            return problems
        try:
            paths, adapter = self._parts(spec)
            self._params(spec)
        except (LookupError, TypeError, ValueError) as exc:
            return [str(exc).strip("'\"")]
        if adapter.local_status(paths.datasets) != "present":
            problems.append(f"{adapter.card.name} is not available yet: get it in the Dataset Hub")
        return problems

    def _data(self, spec: ExperimentSpec, adapter: TabularAdapter, root: Path, qubits: int) -> Prepared:
        training = spec.training
        assert isinstance(training, QuantumTrainingSpec)
        frame, target = adapter.load_frame(root)
        names = getattr(adapter, "class_names", lambda: ())()
        return prepare(frame, target, qubits, seed=spec.seed, test_fraction=training.test_fraction,
                       val_fraction=spec.data.split.val_fraction, subset=spec.data.subset, class_names=names)

    # Run -----------------------------------------------------------------------------------
    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        started = time.monotonic()
        paths, adapter = self._parts(spec)
        params = self._params(spec)
        data = self._data(spec, adapter, paths.datasets, int(params["qubits"]))
        info = {"model": spec.model.model, "dataset": adapter.card.name, "train_samples": len(data.x_train),
                "val_samples": len(data.x_val), "test_samples": len(data.x_test), "classes": data.classes,
                "angles": data.angles, "kept_variance": data.kept_variance, **params}
        _json(ctx.run_dir / "run_info.json", info)
        ctx.tracker.log_params(info)
        callbacks.on_log(f"{adapter.card.name}: {len(data.x_train)} training, {len(data.x_val)} validation, "
                         f"{len(data.x_test)} test samples; {params['qubits']} qubits from "
                         f"{', '.join(data.angles)}")
        try:
            if spec.model.model == "vqc":
                return self._train(spec, params, data, callbacks, ctx, started)
            return self._kernel(spec, params, data, callbacks, ctx, started)
        except Cancelled:
            callbacks.on_log("Cancelled; checkpoints/last.npz resumes from the last finished epoch")
            return RunResult(ctx.run_id, "cancelled", {}, duration_s=time.monotonic() - started)

    def _model(self, params: dict[str, Any], classes: int, seed: int) -> VariationalClassifier:
        return build_vqc(classes=classes, seed=seed, **params)

    def _train(self, spec: ExperimentSpec, params: dict[str, Any], data: Prepared, callbacks: RunCallbacks,
               ctx: RunContext, started: float) -> RunResult:
        training = spec.training
        assert isinstance(training, QuantumTrainingSpec)
        model = self._model(params, len(data.classes), spec.seed)
        optimizer = Adam(training.lr)
        folder = ctx.run_dir / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        first, best_acc = 1, -1.0
        if ctx.resume_from is not None:
            saved = np.load(ctx.resume_from)
            model.params = saved["params"]
            optimizer._m, optimizer._v, optimizer._t = saved["m"], saved["v"], int(saved["t"])
            first, best_acc = int(saved["epoch"]) + 1, float(saved["best"])
            callbacks.on_log(f"Resuming after epoch {first - 1}")
        steps = -(-len(data.x_train) // training.batch_size)
        for epoch in range(first, training.epochs + 1):
            begun = time.monotonic()
            callbacks.on_epoch_start(epoch, training.epochs)
            rng = np.random.default_rng([spec.seed, epoch])
            loss = model.fit_epoch(data.x_train, data.y_train, optimizer, training.batch_size, rng,
                                   cancel=ctx.cancel.raise_if_cancelled)
            callbacks.on_step(steps, steps, {"train_loss": loss})
            metrics = {"train_loss": loss,
                       "train_acc": float((model.predict(data.x_train) == data.y_train).mean())}
            if len(data.x_val):
                metrics.update(val_loss=model.loss(data.x_val, data.y_val),
                               val_acc=float((model.predict(data.x_val) == data.y_val).mean()))
            metrics.update(lr=training.lr, epoch_time_s=time.monotonic() - begun)
            score = metrics.get("val_acc", metrics["train_acc"])
            improved = score > best_acc
            best_acc = max(best_acc, score)
            state = {"params": model.params, "m": optimizer._m, "v": optimizer._v, "t": optimizer._t,
                     "epoch": epoch, "best": best_acc}
            np.savez(folder / "last.npz", **state)
            if improved:
                np.savez(folder / "best.npz", **state)
            callbacks.on_epoch_end(epoch, metrics)
            ctx.tracker.log_metrics(metrics, step=epoch, epoch=epoch)
            _append(ctx.run_dir / "metrics.jsonl", {"epoch": epoch, **metrics})
        model.params = np.load(folder / "best.npz")["params"]
        results = self._score(model.predict_proba(data.x_test), model.predict(data.x_test), data)
        results["circuit"] = model.describe()
        results["boundary"] = self._boundary(data, lambda grid: model.predict_proba(grid))
        return self._finish(spec, results, ctx, callbacks, started, best=folder / "best.npz",
                            extra={"best_val_acc": best_acc, "epochs_completed": float(training.epochs)})

    def _kernel(self, spec: ExperimentSpec, params: dict[str, Any], data: Prepared, callbacks: RunCallbacks,
                ctx: RunContext, started: float) -> RunResult:
        from sklearn.svm import SVC

        callbacks.on_epoch_start(1, 1)
        x_fit = np.vstack([data.x_train, data.x_val]) if len(data.x_val) else data.x_train
        y_fit = np.concatenate([data.y_train, data.y_val]) if len(data.y_val) else data.y_train
        encoding = params["encoding"]
        callbacks.on_log(f"Kernel matrix of {len(x_fit)} training samples ({encoding} feature map)")
        gram = fidelity_kernel(x_fit, encoding=encoding)
        ctx.cancel.raise_if_cancelled()
        svm = SVC(kernel="precomputed", C=float(params["C"]), probability=True, random_state=spec.seed)
        svm.fit(gram, y_fit)
        test_kernel = fidelity_kernel(data.x_test, x_fit, encoding=encoding)
        results = self._score(svm.predict_proba(test_kernel), svm.predict(test_kernel), data)
        order = np.argsort(y_fit, kind="stable")[:: max(1, len(y_fit) // 60)]
        results["kernel"] = {"matrix": gram[np.ix_(order, order)].round(4).tolist(),
                             "labels": y_fit[order].tolist(), "alignment": kernel_alignment(gram, y_fit),
                             "encoding": encoding}
        results["boundary"] = self._boundary(
            data, lambda grid: svm.predict_proba(fidelity_kernel(grid, x_fit, encoding=encoding)))
        metrics = {"train_acc": float((svm.predict(gram) == y_fit).mean()),
                   "kernel_alignment": results["kernel"]["alignment"]}
        folder = ctx.run_dir / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "best.npz", support=svm.support_, dual=svm.dual_coef_, intercept=svm.intercept_,
                 x_fit=x_fit, y_fit=y_fit)
        callbacks.on_epoch_end(1, metrics)
        _append(ctx.run_dir / "metrics.jsonl", {"epoch": 1, **metrics})
        return self._finish(spec, results, ctx, callbacks, started, best=folder / "best.npz",
                            extra={**metrics, "epochs_completed": 1.0})

    # Results -------------------------------------------------------------------------------
    @staticmethod
    def _score(proba: np.ndarray, predicted: np.ndarray, data: Prepared) -> dict[str, Any]:
        from labs.classical_ml.report import classification_report

        report = classification_report(data.y_test, predicted.astype(np.int64), proba, data.classes)
        report["angles"], report["kept_variance"] = data.angles, data.kept_variance
        return report

    @staticmethod
    def _boundary(data: Prepared, proba: Any) -> dict[str, Any] | None:
        """For data with two columns: the predicted class over a grid of the angle plane, and the
        test points, for the decision-boundary view."""
        if len(data.columns) != 2 or data.angles[:2] != data.columns[:2]:
            return None
        axis = np.linspace(0, np.pi, _GRID)
        gx, gy = np.meshgrid(axis, axis)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        qubits = len(data.angles)
        grid = grid[:, [i % 2 for i in range(qubits)]]
        p = np.asarray(proba(grid))
        return {"size": _GRID, "classes": p.argmax(axis=1).reshape(_GRID, _GRID).tolist(),
                "confidence": p.max(axis=1).reshape(_GRID, _GRID).round(3).tolist(),
                "points": [[float(a), float(b), int(c)] for (a, b, *_), c in zip(data.x_test, data.y_test,
                                                                                 strict=True)],
                "axes": data.columns[:2]}

    def _finish(self, spec: ExperimentSpec, results: dict[str, Any], ctx: RunContext, callbacks: RunCallbacks,
                started: float, *, best: Path, extra: dict[str, float]) -> RunResult:
        metrics = {f"test_{key}": value for key, value in results.pop("metrics").items()}
        results.update({"task": spec.task.value, "model": spec.model.model, "metrics": metrics})
        callbacks.on_artifact(_json(ctx.run_dir / "results.json", results), "results")
        callbacks.on_artifact(_json(ctx.run_dir / "test_confusion.json",
                                    {"classes": results["classes"], "matrix": results["matrix"]}),
                              "confusion_matrix")
        callbacks.on_log(f"Test accuracy {metrics['test_acc']:.4f} · macro F1 {metrics['test_f1']:.4f}")
        return RunResult(ctx.run_id, "completed", {**metrics, **extra}, best, time.monotonic() - started)

    # Re-scoring ----------------------------------------------------------------------------
    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        paths, adapter = self._parts(spec)
        params = self._params(spec)
        data = self._data(spec, adapter, paths.datasets, int(params["qubits"]))
        x, y = (data.x_test, data.y_test) if split == "test" else (data.x_val, data.y_val)
        if not len(x):
            raise ValueError(f"this run has no {split} split")
        saved = np.load(checkpoint)
        if spec.model.model == "vqc":
            model = self._model(params, len(data.classes), spec.seed)
            model.params = saved["params"]
            proba = model.predict_proba(x)
        else:
            from sklearn.svm import SVC

            svm = SVC(kernel="precomputed", C=float(params["C"]), probability=True, random_state=spec.seed)
            svm.fit(fidelity_kernel(saved["x_fit"], encoding=params["encoding"]), saved["y_fit"])
            proba = svm.predict_proba(fidelity_kernel(x, saved["x_fit"], encoding=params["encoding"]))
        callbacks.on_log(f"Scoring {checkpoint.name} on {len(x)} {split} samples")
        data.y_test = y
        report = self._score(proba, proba.argmax(axis=1), data)
        return {key: report[key] for key in ("metrics", "classes", "matrix")}
