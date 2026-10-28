"""Quantum kernels: the similarity of two samples is the overlap of the states a feature map puts
them in, ``k(x, x') = |⟨φ(x)|φ(x')⟩|²`` (a fidelity kernel). A support-vector machine on the
precomputed kernel matrix then classifies — the kernel is quantum, the optimiser classical."""

from __future__ import annotations

import numpy as np

from .sim import encode

__all__ = ["fidelity_kernel", "kernel_alignment"]


def fidelity_kernel(a: np.ndarray, b: np.ndarray | None = None, *, encoding: str = "zz") -> np.ndarray:
    """``len(a) × len(b)`` kernel matrix (``b`` defaults to ``a``)."""
    sa = encode(a, encoding)
    sb = sa if b is None else encode(b, encoding)
    return np.abs(sa.conj() @ sb.T) ** 2


def kernel_alignment(kernel: np.ndarray, y: np.ndarray) -> float:
    """How well a kernel matrix matches the labels (centred kernel-target alignment, −1..1)."""
    labels = np.where(y[:, None] == y[None, :], 1.0, -1.0)
    n = len(y)
    centre = np.eye(n) - 1.0 / n
    k, t = centre @ kernel @ centre, centre @ labels @ centre
    denominator = np.linalg.norm(k) * np.linalg.norm(t)
    return float(np.sum(k * t) / denominator) if denominator else 0.0
