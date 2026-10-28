"""A state-vector simulator for whole batches: every sample's state is one row of a
``(batch, 2**n)`` array, so a circuit whose angles depend on the data runs for all samples at
once. Qubit 0 is the most significant bit, as in the Simulation Lab's circuits.

Feature maps put a sample into a state (``angle``: one RY per feature; ``zz``: the second-order
Pauli-Z map of Havlíček et al., Nature 567, 2019), and the hardware-efficient ansatz adds
trainable RY and RZ rotations with a ring of CNOTs per layer. Everything is numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ENCODINGS", "Op", "ansatz", "apply_cnot", "apply_single", "encode", "expectation_z",
           "feature_ops", "run_ops", "ry", "rz", "states"]

ENCODINGS = {"angle": "Angle (one RY per feature)", "zz": "ZZ feature map (Havlíček et al.)"}
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@dataclass(frozen=True)
class Op:
    """One gate: ``RY``/``RZ`` with an angle (per sample or shared), ``H``, or ``CNOT``. The
    ``label`` names where the angle comes from, for drawing: ``x0``, ``θ3``..."""
    gate: str
    qubits: tuple[int, ...]
    label: str = ""


def ry(theta: np.ndarray) -> np.ndarray:
    """RY(θ) for every θ: shape ``(..., 2, 2)``."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def rz(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    zero = np.zeros_like(theta)
    minus, plus = np.exp(-0.5j * theta), np.exp(0.5j * theta)
    return np.stack([np.stack([minus, zero], -1), np.stack([zero, plus], -1)], -2)


def states(batch: int, qubits: int) -> np.ndarray:
    psi = np.zeros((batch, 2 ** qubits), dtype=complex)
    psi[:, 0] = 1.0
    return psi


def apply_single(psi: np.ndarray, qubits: int, qubit: int, matrix: np.ndarray) -> np.ndarray:
    """``matrix`` is ``(2, 2)`` for every sample or ``(batch, 2, 2)``, one per sample."""
    batch = psi.shape[0]
    view = psi.reshape(batch, 2 ** qubit, 2, 2 ** (qubits - qubit - 1))
    if matrix.ndim == 2:
        out = np.einsum("ij,bajc->baic", matrix, view)
    else:
        out = np.einsum("bij,bajc->baic", matrix, view)
    return out.reshape(batch, -1)


def apply_cnot(psi: np.ndarray, qubits: int, control: int, target: int) -> np.ndarray:
    batch = psi.shape[0]
    view = psi.reshape((batch,) + (2,) * qubits).copy()
    index = [slice(None)] * (qubits + 1)
    index[1 + control] = 1
    sub = view[tuple(index)]
    axis = target - (target > control)
    view[tuple(index)] = np.flip(sub, axis=1 + axis)
    return view.reshape(batch, -1)


def feature_ops(qubits: int, encoding: str) -> list[Op]:
    """The encoding circuit as gates; angles named ``x<i>`` (or ``x<i>x<j>`` for ZZ pairs)."""
    if encoding == "angle":
        return [Op("RY", (q,), f"x{q}") for q in range(qubits)]
    if encoding == "zz":
        ops = [Op("H", (q,)) for q in range(qubits)] + [Op("RZ", (q,), f"x{q}") for q in range(qubits)]
        for a in range(qubits):
            for b in range(a + 1, qubits):
                ops += [Op("CNOT", (a, b)), Op("RZ", (b,), f"x{a}x{b}"), Op("CNOT", (a, b))]
        return ops
    raise ValueError(f"unknown encoding {encoding!r} (known: {', '.join(ENCODINGS)})")


def ansatz(qubits: int, layers: int) -> list[Op]:
    """Trainable layers; parameter ``θk`` is entry ``k`` of the flat parameter vector."""
    ops, k = [], 0
    for _ in range(layers):
        for q in range(qubits):
            ops += [Op("RY", (q,), f"θ{k}"), Op("RZ", (q,), f"θ{k + 1}")]
            k += 2
        if qubits > 1:
            ops += [Op("CNOT", (q, (q + 1) % qubits)) for q in range(qubits if qubits > 2 else 1)]
    return ops


def _angle(label: str, x: np.ndarray, params: np.ndarray) -> np.ndarray | float:
    if label.startswith("θ"):
        return float(params[int(label[1:])])
    if "x" in label[1:]:  # a ZZ pair: (π - x_a)(π - x_b)
        a, b = (int(part) for part in label[1:].split("x"))
        return 2 * (np.pi - x[:, a]) * (np.pi - x[:, b])
    return x[:, int(label[1:])]


def run_ops(psi: np.ndarray, qubits: int, ops: list[Op], x: np.ndarray, params: np.ndarray,
            *, zz: bool = False) -> np.ndarray:
    """Apply ``ops`` to the batch; data angles come from ``x`` (``batch × features``)."""
    for op in ops:
        if op.gate == "CNOT":
            psi = apply_cnot(psi, qubits, *op.qubits)
        elif op.gate == "H":
            psi = apply_single(psi, qubits, op.qubits[0], _H)
        else:
            angle = _angle(op.label, x, params)
            if zz and op.label.startswith("x") and "x" not in op.label[1:]:
                angle = 2 * np.asarray(angle)  # the ZZ map's single-qubit terms are RZ(2x)
            matrix = (ry if op.gate == "RY" else rz)(np.asarray(angle))
            psi = apply_single(psi, qubits, op.qubits[0], matrix)
    return psi


def encode(x: np.ndarray, encoding: str) -> np.ndarray:
    """The state of every sample after the feature map."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    qubits = x.shape[1]
    return run_ops(states(len(x), qubits), qubits, feature_ops(qubits, encoding), x, np.zeros(0),
                   zz=encoding == "zz")


def expectation_z(psi: np.ndarray, qubits: int) -> np.ndarray:
    """⟨Z_q⟩ for every sample and qubit: ``batch × qubits``."""
    probs = np.abs(psi) ** 2
    out = np.empty((psi.shape[0], qubits))
    for q in range(qubits):
        view = probs.reshape(psi.shape[0], 2 ** q, 2, -1)
        out[:, q] = view[:, :, 0, :].sum(axis=(1, 2)) - view[:, :, 1, :].sum(axis=(1, 2))
    return out
