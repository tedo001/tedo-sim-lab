"""Exact simulation of a circuit's state vector (numpy; up to ``MAX_QUBITS`` qubits).

The state is a tensor with one axis per qubit, qubit 0 first, so basis index ``i`` reads as the
bit string ``|q0 q1 … ⟩``. Also: outcome probabilities, sampled shots (seeded), each qubit's
Bloch vector and purity (how entangled it is with the rest), and Pauli expectation values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit, Op
from .gates import gate

__all__ = ["Result", "apply", "bloch_vector", "expectation", "probabilities", "sample", "simulate",
           "unitary_of"]


def zero_state(qubits: int) -> np.ndarray:
    state = np.zeros(2 ** qubits, dtype=complex)
    state[0] = 1.0
    return state


def apply(state: np.ndarray, op: Op, qubits: int) -> np.ndarray:
    """``op`` applied to ``state`` (a new array)."""
    matrix = gate(op.gate).matrix(*op.params)
    psi = state.reshape((2,) * qubits).copy()
    k = len(op.targets)
    index = [slice(None)] * qubits
    for control in op.controls:
        index[control] = 1
    view = psi[tuple(index)]  # the part where every control is |1⟩
    remaining = [q for q in range(qubits) if q not in op.controls]
    axes = [remaining.index(t) for t in op.targets]
    tensor = matrix.reshape((2,) * (2 * k))
    moved = np.tensordot(tensor, view, axes=(list(range(k, 2 * k)), axes))
    view[...] = np.moveaxis(moved, list(range(k)), axes)
    return psi.reshape(-1)


@dataclass(frozen=True)
class Result:
    qubits: int
    state: np.ndarray

    @property
    def probabilities(self) -> np.ndarray:
        return probabilities(self.state)

    def label(self, index: int) -> str:
        return format(index, f"0{self.qubits}b")

    def amplitudes(self, threshold: float = 1e-12) -> list[tuple[str, complex, float]]:
        """(basis label, amplitude, probability) for every basis state that is not empty."""
        probs = self.probabilities
        return [(self.label(i), complex(self.state[i]), float(probs[i]))
                for i in range(len(self.state)) if probs[i] > threshold]

    def bloch(self, qubit: int) -> tuple[float, float, float]:
        return bloch_vector(self.state, qubit, self.qubits)


def simulate(circuit: Circuit, state: np.ndarray | None = None) -> Result:
    psi = zero_state(circuit.qubits) if state is None else np.asarray(state, dtype=complex)
    for op in circuit.ordered():
        psi = apply(psi, op, circuit.qubits)
    return Result(circuit.qubits, psi)


def probabilities(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def sample(state: np.ndarray, shots: int, seed: int = 0) -> dict[str, int]:
    """Measurement counts from ``shots`` runs (reproducible with ``seed``), most frequent first."""
    qubits = int(np.log2(len(state)))
    outcomes = np.random.default_rng(seed).choice(len(state), size=max(1, shots), p=probabilities(state))
    values, counts = np.unique(outcomes, return_counts=True)
    found = {format(int(v), f"0{qubits}b"): int(c) for v, c in zip(values, counts, strict=True)}
    return dict(sorted(found.items(), key=lambda item: (-item[1], item[0])))


def reduced_density(state: np.ndarray, qubit: int, qubits: int) -> np.ndarray:
    psi = np.moveaxis(state.reshape((2,) * qubits), qubit, 0).reshape(2, -1)
    return psi @ psi.conj().T


def bloch_vector(state: np.ndarray, qubit: int, qubits: int) -> tuple[float, float, float]:
    """(x, y, z) of one qubit; shorter than 1 when it is entangled with the others."""
    rho = reduced_density(state, qubit, qubits)
    return (float(2 * rho[0, 1].real), float(-2 * rho[0, 1].imag), float((rho[0, 0] - rho[1, 1]).real))


def purity(state: np.ndarray, qubit: int, qubits: int) -> float:
    """Tr ρ² of one qubit: 1 when it is on its own, ½ when maximally entangled."""
    rho = reduced_density(state, qubit, qubits)
    return float(np.real(np.trace(rho @ rho)))


_PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]])}


def expectation(state: np.ndarray, pauli: str) -> float:
    """⟨ψ|P|ψ⟩ for a Pauli string such as ``"ZZI"`` (one letter per qubit, qubit 0 first)."""
    qubits = int(np.log2(len(state)))
    if len(pauli) != qubits or any(p not in _PAULI for p in pauli.upper()):
        raise ValueError(f"expected {qubits} letters from I, X, Y, Z")
    psi = state.reshape((2,) * qubits)
    for qubit, letter in enumerate(pauli.upper()):
        if letter != "I":
            psi = np.moveaxis(np.tensordot(_PAULI[letter], psi, axes=([1], [qubit])), 0, qubit)
    return float(np.real(np.vdot(state, psi.reshape(-1))))


def unitary_of(circuit: Circuit) -> np.ndarray:
    """The circuit's matrix (2ⁿ × 2ⁿ), column j = the circuit applied to basis state j."""
    size = 2 ** circuit.qubits
    columns = [simulate(circuit, np.eye(size, dtype=complex)[j]).state for j in range(size)]
    return np.stack(columns, axis=1)
