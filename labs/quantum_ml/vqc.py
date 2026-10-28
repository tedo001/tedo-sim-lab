"""A variational quantum classifier: a feature map, trainable layers, and ⟨Z⟩ on the first qubits
as class scores. Gradients come from the parameter-shift rule — for RY and RZ the exact
derivative of an expectation is ``(E(θ + π/2) − E(θ − π/2)) / 2`` — so training uses only circuit
runs, as it would on quantum hardware. Optimised with Adam on the cross-entropy loss.

With ``classes`` classes (at most one per qubit), class ``c`` scores ``−scale · ⟨Z_c⟩``: a qubit
pushed towards |1⟩ votes for its class. Two classes use qubits 0 and 1, so a single-qubit
circuit cannot separate two classes and is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .sim import Op, ansatz, expectation_z, feature_ops, run_ops, states

__all__ = ["Adam", "VariationalClassifier"]

SHIFT = np.pi / 2


@dataclass
class Adam:
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: np.ndarray | None = field(default=None, repr=False)
    _v: np.ndarray | None = field(default=None, repr=False)
    _t: int = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m, self._v = np.zeros_like(params), np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class VariationalClassifier:
    def __init__(self, qubits: int, classes: int, *, layers: int = 2, encoding: str = "angle",
                 scale: float = 4.0, seed: int = 0) -> None:
        if qubits < 2 or classes < 2 or classes > qubits:
            raise ValueError(f"{classes} classes need between {max(2, classes)} and 10 qubits (one per class "
                             f"score); got {qubits}")
        if qubits > 10:
            raise ValueError("at most 10 qubits")
        self.qubits, self.classes, self.layers = qubits, classes, layers
        self.encoding, self.scale = encoding, scale
        self.ops: list[Op] = feature_ops(qubits, encoding) + ansatz(qubits, layers)
        self.n_params = 2 * qubits * layers
        self.params = np.random.default_rng(seed).normal(0.0, 0.1, self.n_params)

    # Forward ---------------------------------------------------------------------------------
    def _expect(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        psi = run_ops(states(len(x), self.qubits), self.qubits, self.ops, x, params,
                      zz=self.encoding == "zz")
        return expectation_z(psi, self.qubits)[:, :self.classes]

    def scores(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        return -self.scale * self._expect(np.atleast_2d(x), self.params if params is None else params)

    def predict_proba(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        logits = self.scores(x, params)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def loss(self, x: np.ndarray, y: np.ndarray, params: np.ndarray | None = None) -> float:
        p = self.predict_proba(x, params)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    # Gradient ----------------------------------------------------------------------------------
    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """(loss, ∂loss/∂θ) by the parameter-shift rule: 2 circuit runs per parameter."""
        p = self.predict_proba(x)
        onehot = np.eye(self.classes)[y]
        dloss_dlogit = (p - onehot) / len(x)                     # batch × classes
        grad = np.empty(self.n_params)
        for k in range(self.n_params):
            plus, minus = self.params.copy(), self.params.copy()
            plus[k] += SHIFT
            minus[k] -= SHIFT
            dexp = (self._expect(x, plus) - self._expect(x, minus)) / 2   # batch × classes
            grad[k] = np.sum(dloss_dlogit * (-self.scale) * dexp)
        loss = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        return loss, grad

    def fit_epoch(self, x: np.ndarray, y: np.ndarray, optimizer: Adam, batch_size: int,
                  rng: np.random.Generator, cancel: Callable[[], None] | None = None) -> float:
        """One pass over shuffled batches; returns the mean training loss of the epoch."""
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(x), batch_size):
            if cancel is not None:
                cancel()
            batch = order[start:start + batch_size]
            loss, grad = self.gradient(x[batch], y[batch])
            self.params = optimizer.step(self.params, grad)
            losses.append(loss * len(batch))
        return float(np.sum(losses) / len(x))

    def describe(self) -> dict[str, object]:
        return {"qubits": self.qubits, "classes": self.classes, "layers": self.layers,
                "encoding": self.encoding, "parameters": self.n_params,
                "gates": [[op.gate, list(op.qubits), op.label] for op in self.ops]}
