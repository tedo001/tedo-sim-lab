"""Quantum kernels: the similarity of two samples is the overlap of the states a feature map puts
them in, ``k(x, x') = |⟨φ(x)|φ(x')⟩|²`` (a fidelity kernel). A support-vector machine on the
precomputed kernel matrix then classifies — the kernel is quantum, the optimiser classical."""

from __future__ import annotations

from typing import Any

import numpy as np

from .sim import encode, feature_ops

__all__ = ["fidelity_kernel", "kernel_alignment"]


def fidelity_kernel(a: np.ndarray, b: np.ndarray | None = None, *, encoding: str = "zz",
                    backend: Any = None) -> np.ndarray:
    """``len(a) × len(b)`` kernel matrix (``b`` defaults to ``a``). With a noisy ``backend`` the
    states are density matrices and the kernel is Tr(ρ(x) ρ(x')); with shots it is estimated."""
    if backend is not None and not backend.exact and backend.noise is not None and not backend.noise.ideal:
        a = np.atleast_2d(np.asarray(a, dtype=float))
        qubits = a.shape[1]
        ops = feature_ops(qubits, encoding)
        ra = backend.density(qubits, ops, a, np.zeros(0), zz=encoding == "zz")
        rb = ra if b is None else backend.density(qubits, ops, np.atleast_2d(np.asarray(b, dtype=float)),
                                                  np.zeros(0), zz=encoding == "zz")
        return backend.kernel(ra, rb)
    sa = encode(a, encoding)
    sb = sa if b is None else encode(b, encoding)
    gram = np.abs(sa.conj() @ sb.T) ** 2
    if backend is not None and backend.shots:
        gram = backend.rng.binomial(backend.shots, np.clip(gram, 0, 1)) / backend.shots
        backend.shots_used += backend.shots * gram.size
    return gram


def kernel_alignment(kernel: np.ndarray, y: np.ndarray) -> float:
    """How well a kernel matrix matches the labels (centred kernel-target alignment, −1..1)."""
    labels = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    n = len(y)
    centre = np.eye(n) - 1.0 / n
    k, t = centre @ kernel @ centre, centre @ labels @ centre
    denominator = np.linalg.norm(k) * np.linalg.norm(t)
    return float(np.sum(k * t) / denominator) if denominator else 0.0
