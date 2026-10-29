"""The evaluation protocol of a quantum classifier run, written into its ``results.json`` as
``protocol``:

- **classical baselines** fitted on exactly the same training rows and features (the angles the
  circuit sees) and scored on the same test rows;
- **resources**: qubits, circuit depth, one- and two-qubit gate counts, trainable parameters,
  circuit evaluations, shots, the memory one simulated state takes, and the wall time;
- **simulator settings**: ideal state vector or noisy density matrix, every noise parameter,
  shots, and the seed;
- **noise comparison**: the trained model's test accuracy with the noise scaled 0×, ½×, 1× and 2×
  (the run's own noise, or, for an ideal run, an illustrative reference device), so its sensitivity
  to noise is visible;
- **claims**: simulated, not hardware; and a plain statement of how the model compares with the
  classical baselines. No result here is evidence of quantum advantage: that needs hardware, scaling
  with problem size, and the best classical methods, none of which a small simulation provides."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from labs.simulation.quantum.noise import REFERENCE, NoiseModel

from .noisy import Backend
from .sim import Op

__all__ = ["BASELINES", "FACTORS", "baselines", "claims", "evaluate", "noise_comparison", "resources"]

FACTORS = (0.0, 0.5, 1.0, 2.0)
BASELINES = {"logistic_regression": "Logistic regression", "svm_rbf": "SVM (RBF kernel)",
             "mlp": "Small neural network (16 hidden units)"}


def _classical(name: str, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000, random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")


def _parameters(model: Any) -> int | None:
    if hasattr(model, "coef_"):
        return int(np.size(model.coef_) + np.size(model.intercept_))
    if hasattr(model, "coefs_"):
        return int(sum(np.size(w) for w in model.coefs_) + sum(np.size(b) for b in model.intercepts_))
    if hasattr(model, "support_vectors_"):
        return int(np.size(model.dual_coef_) + np.size(model.support_vectors_))
    return None


def baselines(names: tuple[str, ...], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
              y_test: np.ndarray, seed: int) -> list[dict[str, Any]]:
    """Each classical model on the same split and features: test accuracy, macro F1, fit time."""
    from sklearn.metrics import f1_score

    out = []
    for name in names:
        model = _classical(name, seed)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit = time.perf_counter() - started
        predicted = model.predict(x_test)
        out.append({"model": name, "title": BASELINES[name], "test_acc": float((predicted == y_test).mean()),
                    "test_f1": float(f1_score(y_test, predicted, average="macro")), "fit_time_s": fit,
                    "parameters": _parameters(model)})
    return out


def _depth(ops: list[Op], qubits: int) -> int:
    level = [0] * qubits
    for op in ops:
        top = max(level[q] for q in op.qubits) + 1
        for q in op.qubits:
            level[q] = top
    return max(level, default=0)


def resources(ops: list[Op], qubits: int, parameters: int, backend: Backend, evaluations: int,
              wall_time_s: float) -> dict[str, Any]:
    noisy = backend.noise is not None and not backend.noise.ideal
    state_bytes = 16 * (4 ** qubits if noisy else 2 ** qubits)
    return {"qubits": qubits, "depth": _depth(ops, qubits),
            "gates_1q": sum(1 for op in ops if len(op.qubits) == 1),
            "gates_2q": sum(1 for op in ops if len(op.qubits) == 2), "parameters": parameters,
            "circuit_evaluations": int(evaluations), "shots": int(backend.shots_used),
            "state_memory_bytes": state_bytes, "wall_time_s": wall_time_s}


def noise_comparison(score: Callable[[Backend], float], backend: Backend, seed: int) -> dict[str, Any]:
    """``score(backend)`` (test accuracy) with the run's noise — or the reference device for an
    ideal run — scaled by each of :data:`FACTORS`."""
    own = backend.noise is not None and not backend.noise.ideal
    base: NoiseModel = backend.noise if own else REFERENCE
    rows = []
    for factor in FACTORS:
        trial = Backend(base.scaled(factor), backend.shots, seed)
        rows.append({"factor": factor, "test_acc": float(score(trial)), "noise": trial.noise.describe()})
    return {"reference": "the run's own noise model" if own else
            "an illustrative reference device (not a calibration of any real device)",
            "base": base.describe(), "rows": rows}


def claims(test_acc: float, classical: list[dict[str, Any]], backend: Backend,
           test_samples: int) -> dict[str, Any]:
    best = max(classical, key=lambda b: b["test_acc"]) if classical else None
    if best is None:
        comparison = "No classical baseline was fitted, so the result cannot be put in context."
    elif test_acc <= best["test_acc"]:
        comparison = (f"The classical {best['title'].lower()} does as well or better on the same split "
                      f"({best['test_acc']:.3f} against {test_acc:.3f}).")
    else:
        comparison = (f"Higher than the best classical baseline here ({test_acc:.3f} against "
                      f"{best['test_acc']:.3f} for the {best['title'].lower()}) on {test_samples} test "
                      "samples; a difference this size on one small split and one seed is not evidence "
                      "of anything general — run a seed study and compare intervals.")
    return {"hardware": False, "backend": f"classical simulation ({backend.kind})",
            "results_are": "simulator results, not hardware results",
            "quantum_advantage": "not claimed: a small classical simulation cannot show it (it needs quantum "
                                 "hardware, scaling with problem size and the best classical methods)",
            "comparison": comparison}


def evaluate(training: Any, data: Any, backend: Backend, *, ops: list[Op], qubits: int, parameters: int,
             evaluations: int, started: float, test_acc: float, score_with: Callable[[Backend], float],
             seed: int) -> dict[str, Any]:
    """Everything above for one finished run (``score_with`` re-scores the trained model on another
    backend, for the noise comparison)."""
    wall = time.monotonic() - started
    classical = baselines(tuple(training.baselines), data.x_train, data.y_train, data.x_test, data.y_test,
                          seed)
    return {"simulator": {**backend.describe(), "mode": training.simulator.mode},
            "resources": resources(ops, qubits, parameters, backend, evaluations, wall),
            "baselines": classical, "noise_comparison": noise_comparison(score_with, backend, seed),
            "split": {"train": len(data.x_train), "val": len(data.x_val), "test": len(data.x_test),
                      "features": data.angles, "same_split_for_baselines": True},
            "claims": claims(test_acc, classical, backend, len(data.x_test))}
