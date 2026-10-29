"""Noisy simulation of circuits with density matrices: what an imperfect device does to a state.

A :class:`NoiseModel` adds, after every gate, on the qubits the gate touched:

- **depolarizing noise** with probability λ (``depolarizing_1q`` after one-qubit gates,
  ``depolarizing_2q`` after gates on two or more qubits): ρ → (1 − λ)ρ + λ · I/2ᵏ ⊗ Tr_qubits ρ, the
  convention of Qiskit's ``depolarizing_error``;
- **thermal relaxation** for the gate's duration ``t`` (``gate_time_1q_ns`` / ``gate_time_2q_ns``):
  amplitude damping with γ = 1 − exp(−t/T1), then pure dephasing so coherences decay as exp(−t/T2)
  in total (T2 ≤ 2·T1); idle qubits are not damped;

and, when reading out, a symmetric **readout error** (each measured bit flips with probability
``readout_error``) and a finite number of **shots**. Everything is numpy; density matrices grow
as 4ⁿ, so noisy simulation stops at ``MAX_NOISY_QUBITS``.

The default numbers of :data:`REFERENCE` are illustrative (of the order published for current
superconducting processors), not a calibration of any device; results under it are simulations,
never hardware results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["MAX_NOISY_QUBITS", "NoiseModel", "REFERENCE", "apply_kraus", "apply_matrix", "apply_noise",
           "bloch_of", "depolarizing_kraus", "fidelity_to_pure", "readout_probabilities", "reduced_qubit",
           "relaxation_kraus", "sample_counts", "simulate_noisy"]

MAX_NOISY_QUBITS = 8
_PAULIS = (np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex),
           np.array([[0, -1j], [1j, 0]], dtype=complex), np.array([[1, 0], [0, -1]], dtype=complex))


@dataclass(frozen=True)
class NoiseModel:
    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    #: Relaxation and coherence times in microseconds; ``None`` = no relaxation.
    t1_us: float | None = None
    t2_us: float | None = None
    gate_time_1q_ns: float = 50.0
    gate_time_2q_ns: float = 300.0
    readout_error: float = 0.0

    def __post_init__(self) -> None:
        for name in ("depolarizing_1q", "depolarizing_2q"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} is a probability between 0 and 1")
        if not 0 <= self.readout_error <= 0.5:
            raise ValueError("readout_error is between 0 and 0.5")
        if (self.t1_us is None) != (self.t2_us is None):
            raise ValueError("give T1 and T2 together")
        if self.t1_us is not None and (self.t1_us <= 0 or self.t2_us <= 0 or self.t2_us > 2 * self.t1_us):
            raise ValueError("T1 and T2 are positive, and T2 can be at most 2·T1")

    @property
    def ideal(self) -> bool:
        return (not self.depolarizing_1q and not self.depolarizing_2q and self.t1_us is None
                and not self.readout_error)

    def scaled(self, factor: float) -> NoiseModel:
        """The same device ``factor`` times as noisy (error rates × factor, times ÷ factor)."""
        if factor == 0:
            return NoiseModel()
        return NoiseModel(min(self.depolarizing_1q * factor, 1.0), min(self.depolarizing_2q * factor, 1.0),
                          None if self.t1_us is None else self.t1_us / factor,
                          None if self.t2_us is None else self.t2_us / factor,
                          self.gate_time_1q_ns, self.gate_time_2q_ns, min(self.readout_error * factor, 0.5))

    def describe(self) -> dict[str, object]:
        return asdict(self)


#: Illustrative device-like noise (see the module notes); used to show how sensitive an ideally
#: trained circuit is, never presented as a device.
REFERENCE = NoiseModel(depolarizing_1q=0.001, depolarizing_2q=0.01, t1_us=100.0, t2_us=80.0,
                       gate_time_1q_ns=50.0, gate_time_2q_ns=300.0, readout_error=0.02)


def apply_matrix(rho: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """U ρ U† for a batch of density matrices ``(batch, 2ⁿ, 2ⁿ)``; ``matrix`` acts on ``qubits``
    (``(2ᵏ, 2ᵏ)`` shared, or ``(batch, 2, 2)`` one per sample for a single qubit)."""
    batch, k = rho.shape[0], len(qubits)
    if matrix.ndim == 3:  # one matrix per sample, single qubit
        q = qubits[0]
        a, c = 2 ** q, 2 ** (n - q - 1)
        view = rho.reshape(batch, a, 2, c, a, 2, c)
        view = np.einsum("bij,bxjyuvw->bxiyuvw", matrix, view)
        view = np.einsum("bxiyuvw,blv->bxiyulw", view, matrix.conj())
        return view.reshape(batch, 2 ** n, 2 ** n)
    tensor = rho.reshape((batch,) + (2,) * (2 * n))
    gate = matrix.reshape((2,) * (2 * k))
    rows = [1 + q for q in qubits]
    tensor = np.tensordot(gate, tensor, axes=(list(range(k, 2 * k)), rows))  # new row axes first
    tensor = np.moveaxis(tensor, list(range(k)), rows)
    cols = [1 + n + q for q in qubits]
    tensor = np.tensordot(tensor, gate.conj(), axes=(cols, list(range(k, 2 * k))))  # new column axes last
    tensor = np.moveaxis(tensor, list(range(tensor.ndim - k, tensor.ndim)), cols)
    return tensor.reshape(batch, 2 ** n, 2 ** n)


def apply_kraus(rho: np.ndarray, kraus: list[np.ndarray], qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Σ K ρ K† over the channel's Kraus operators."""
    return sum(apply_matrix(rho, k, qubits, n) for k in kraus)


def depolarizing_kraus(lam: float, k: int) -> list[np.ndarray]:
    """(1 − λ)ρ + λ I/2ᵏ Tr ρ as Pauli Kraus operators on ``k`` qubits."""
    ops = [np.eye(1, dtype=complex)]
    for _ in range(k):
        ops = [np.kron(a, p) for a in ops for p in _PAULIS]
    count = len(ops)  # 4ᵏ
    weights = [1 - lam * (count - 1) / count] + [lam / count] * (count - 1)
    return [np.sqrt(w) * op for w, op in zip(weights, ops, strict=True)]


def relaxation_kraus(t_ns: float, t1_us: float, t2_us: float) -> list[list[np.ndarray]]:
    """Amplitude damping then pure dephasing for ``t_ns``: two one-qubit channels."""
    t, t1, t2 = t_ns * 1e-3, t1_us, t2_us
    gamma = 1 - np.exp(-t / t1)
    damping = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
               np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]
    keep = np.exp(-2 * t / t2 + t / t1)  # (1 − λ) so that coherence decays as exp(−t/T2) overall
    lam = float(np.clip(1 - keep, 0, 1))
    dephasing = [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
                 np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]
    return [damping, dephasing]


def apply_noise(rho: np.ndarray, qubits: tuple[int, ...], n: int, noise: NoiseModel) -> np.ndarray:
    """The noise that follows a gate on ``qubits``."""
    if noise.ideal:
        return rho
    two = len(qubits) > 1
    lam = noise.depolarizing_2q if two else noise.depolarizing_1q
    if lam:
        rho = apply_kraus(rho, depolarizing_kraus(lam, len(qubits)), qubits, n)
    if noise.t1_us is not None:
        duration = noise.gate_time_2q_ns if two else noise.gate_time_1q_ns
        for channel in relaxation_kraus(duration, noise.t1_us, noise.t2_us):
            for q in qubits:
                rho = apply_kraus(rho, channel, (q,), n)
    return rho


def readout_probabilities(probabilities: np.ndarray, n: int, error: float) -> np.ndarray:
    """Outcome probabilities after each bit flips with probability ``error`` (``(…, 2ⁿ)``)."""
    if not error:
        return probabilities
    flip = np.array([[1 - error, error], [error, 1 - error]])
    shape = probabilities.shape
    tensor = probabilities.reshape(shape[:-1] + (2,) * n)
    lead = len(shape) - 1
    for q in range(n):
        tensor = np.moveaxis(np.tensordot(flip, tensor, axes=([1], [lead + q])), 0, lead + q)
    return tensor.reshape(shape)


def simulate_noisy(circuit, noise: NoiseModel):  # noqa: ANN001, ANN201 (Circuit → density matrix)
    """The circuit's density matrix under ``noise`` (Simulation Lab circuits; controls included)."""
    from .gates import gate

    n = circuit.qubits
    if n > MAX_NOISY_QUBITS:
        raise ValueError(f"noisy simulation keeps a 4ⁿ density matrix: at most {MAX_NOISY_QUBITS} qubits")
    rho = np.zeros((1, 2 ** n, 2 ** n), dtype=complex)
    rho[0, 0, 0] = 1.0
    for op in circuit.ordered():
        matrix = gate(op.gate).matrix(*op.params)
        involved = tuple(op.controls) + tuple(op.targets)
        if op.controls:  # identity unless every control is |1⟩
            size = 2 ** len(involved)
            full = np.eye(size, dtype=complex)
            block = matrix.shape[0]
            full[size - block:, size - block:] = matrix
            matrix = full
        rho = apply_matrix(rho, matrix, involved, n)
        rho = apply_noise(rho, involved, n, noise)
    return rho[0]


def reduced_qubit(rho: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """One qubit's 2 × 2 density matrix (the others traced out)."""
    tensor = np.moveaxis(rho.reshape((2,) * (2 * n)), (qubit, n + qubit), (0, n))
    tensor = tensor.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.einsum("ajbj->ab", tensor)


def bloch_of(rho1: np.ndarray) -> tuple[float, float, float]:
    return (float(2 * rho1[0, 1].real), float(-2 * rho1[0, 1].imag), float((rho1[0, 0] - rho1[1, 1]).real))


def sample_counts(probabilities: np.ndarray, shots: int, seed: int = 0) -> dict[str, int]:
    """Counts of ``shots`` outcomes drawn from ``probabilities``, most frequent first."""
    p = np.clip(np.real(probabilities), 0, None)
    p = p / p.sum()
    n = int(np.log2(len(p)))
    outcomes = np.random.default_rng(seed).choice(len(p), size=max(1, shots), p=p)
    values, counts = np.unique(outcomes, return_counts=True)
    found = {format(int(v), f"0{n}b"): int(c) for v, c in zip(values, counts, strict=True)}
    return dict(sorted(found.items(), key=lambda item: (-item[1], item[0])))


def fidelity_to_pure(rho: np.ndarray, state: np.ndarray) -> float:
    """⟨ψ|ρ|ψ⟩: how close the noisy state is to the ideal one (1 = identical)."""
    return float(np.real(np.vdot(state, rho @ state)))
