"""Running the Quantum ML circuits on a simulated imperfect device: batched density matrices with
the Simulation Lab's noise channels (:mod:`labs.simulation.quantum.noise`) after every gate, readout
bit flips, and expectations estimated from a finite number of shots. An ideal backend with shots
only samples the exact state's outcomes. Every number this produces is a classical simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from labs.simulation.quantum.noise import MAX_NOISY_QUBITS, NoiseModel, apply_matrix, apply_noise

from .sim import _H, Op, _angle, ry, rz

__all__ = ["Backend", "backend_from_spec"]


def _cnot_permutation(n: int, control: int, target: int) -> np.ndarray:
    index = np.arange(2 ** n)
    on = (index >> (n - 1 - control)) & 1
    return index ^ (on << (n - 1 - target))


@dataclass
class Backend:
    """``noise`` (``None`` = exact) and ``shots`` (``None`` = exact expectations)."""
    noise: NoiseModel | None = None
    shots: int | None = None
    seed: int = 0
    #: Circuit evaluations and shots spent so far (resource accounting).
    evaluations: int = 0
    shots_used: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def exact(self) -> bool:
        return (self.noise is None or self.noise.ideal) and self.shots is None

    @property
    def kind(self) -> str:
        if self.noise is not None and not self.noise.ideal:
            return "density matrix with noise"
        return "state vector" + (" with shots" if self.shots else "")

    def density(self, qubits: int, ops: list[Op], x: np.ndarray, params: np.ndarray, *,
                zz: bool = False) -> np.ndarray:
        """The batch's density matrices after ``ops`` under the noise model."""
        if qubits > MAX_NOISY_QUBITS:
            raise ValueError(f"noisy simulation keeps 4ⁿ density matrices: at most {MAX_NOISY_QUBITS} qubits")
        noise = self.noise or NoiseModel()
        rho = np.zeros((len(x), 2 ** qubits, 2 ** qubits), dtype=complex)
        rho[:, 0, 0] = 1.0
        for op in ops:
            if op.gate == "CNOT":
                perm = _cnot_permutation(qubits, *op.qubits)
                rho = rho[:, perm][:, :, perm]
            elif op.gate == "H":
                rho = apply_matrix(rho, _H, op.qubits, qubits)
            else:
                angle = _angle(op.label, x, params)
                if zz and op.label.startswith("x") and "x" not in op.label[1:]:
                    angle = 2 * np.asarray(angle)
                matrix = (ry if op.gate == "RY" else rz)(np.asarray(angle))
                if matrix.ndim == 3 and matrix.shape[0] != len(x):
                    matrix = np.broadcast_to(matrix, (len(x), 2, 2))
                rho = apply_matrix(rho, matrix, op.qubits, qubits)
            rho = apply_noise(rho, op.qubits, qubits, noise)
        self.evaluations += len(x)
        return rho

    def expect_z(self, qubits: int, ops: list[Op], x: np.ndarray, params: np.ndarray, *,
                 zz: bool = False, exact_state: Any = None) -> np.ndarray:
        """⟨Z_q⟩ per sample and qubit, as a device would report it."""
        if self.noise is not None and not self.noise.ideal:
            probs = np.real(np.einsum("bii->bi", self.density(qubits, ops, x, params, zz=zz)))
        else:
            probs = np.abs(exact_state(x, params)) ** 2
            self.evaluations += len(x)
        out = np.empty((len(x), qubits))
        for q in range(qubits):
            view = probs.reshape(len(x), 2 ** q, 2, -1)
            out[:, q] = view[:, :, 0, :].sum(axis=(1, 2)) - view[:, :, 1, :].sum(axis=(1, 2))
        if self.noise is not None and self.noise.readout_error:
            out *= 1 - 2 * self.noise.readout_error
        if self.shots:
            ones = self.rng.binomial(self.shots, np.clip((1 - out) / 2, 0, 1))
            out = 1 - 2 * ones / self.shots
            self.shots_used += self.shots * len(x)
        return out

    def kernel(self, a_rho: np.ndarray, b_rho: np.ndarray) -> np.ndarray:
        """Tr(ρᵢ ρⱼ) for every pair, estimated from shots when ``shots`` is set."""
        flat_a, flat_b = a_rho.reshape(len(a_rho), -1), b_rho.reshape(len(b_rho), -1)
        gram = np.clip(np.real(flat_a @ flat_b.conj().T), 0, 1)
        if self.shots:
            gram = self.rng.binomial(self.shots, gram) / self.shots
            self.shots_used += self.shots * gram.size
        return gram

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "noise": None if self.noise is None else self.noise.describe(),
                "shots": self.shots, "seed": self.seed}


def backend_from_spec(simulator: Any, seed: int) -> Backend:
    """The backend a :class:`SimulatorSpec` describes."""
    noise = None
    if simulator.mode == "noisy":
        noise = NoiseModel(simulator.depolarizing_1q, simulator.depolarizing_2q, simulator.t1_us,
                           simulator.t2_us, simulator.gate_time_1q_ns, simulator.gate_time_2q_ns,
                           simulator.readout_error)
    return Backend(noise, simulator.shots, seed)
