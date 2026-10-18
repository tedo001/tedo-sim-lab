"""Quantum gates: their matrices and how many qubits they act on.

Matrices are in the computational basis with the first qubit as the most significant bit, the
convention the circuit simulator and the basis labels (``|q0 q1 …⟩``) use. Controlled gates are
their target gate plus control qubits, so CNOT is ``X`` with one control and Toffoli ``X`` with two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GATES", "GateSpec", "gate"]

_S = np.sqrt(0.5)


@dataclass(frozen=True)
class GateSpec:
    name: str
    title: str
    #: Qubits the matrix acts on (1, or 2 for SWAP).
    targets: int = 1
    #: Control qubits (0 for plain gates).
    controls: int = 0
    params: tuple[str, ...] = ()
    #: The matrix acted out on the targets (controls are added by the simulator).
    base: str = ""
    help: str = ""

    def matrix(self, *params: float) -> np.ndarray:
        name = self.base or self.name
        if len(params) != len(self.params):
            raise ValueError(f"{self.name} takes {len(self.params)} parameter(s)")
        return _matrix(name, params)


def _matrix(name: str, params: tuple[float, ...]) -> np.ndarray:
    if name == "I":
        return np.eye(2, dtype=complex)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "H":
        return np.array([[_S, _S], [_S, -_S]], dtype=complex)
    if name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "Sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if name == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    if name == "Tdg":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    if name == "SX":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    if name == "RX":
        (theta,) = params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "RY":
        (theta,) = params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "RZ":
        (theta,) = params
        return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)
    if name == "P":
        (phi,) = params
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    if name == "U":
        theta, phi, lam = params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -np.exp(1j * lam) * s],
                         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
    if name == "SWAP":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise KeyError(f"unknown gate {name!r}")


GATES: dict[str, GateSpec] = {spec.name: spec for spec in (
    GateSpec("X", "Pauli X (NOT)", help="Flips |0⟩ and |1⟩"),
    GateSpec("Y", "Pauli Y", help="A bit and phase flip"),
    GateSpec("Z", "Pauli Z", help="Flips the phase of |1⟩"),
    GateSpec("H", "Hadamard", help="Equal superposition: |0⟩ → |+⟩"),
    GateSpec("S", "S (√Z)", help="Phase of π/2 on |1⟩"),
    GateSpec("Sdg", "S†", help="Phase of −π/2 on |1⟩"),
    GateSpec("T", "T (π/8)", help="Phase of π/4 on |1⟩"),
    GateSpec("Tdg", "T†", help="Phase of −π/4 on |1⟩"),
    GateSpec("SX", "√X", help="Half a NOT"),
    GateSpec("RX", "Rotation about X", params=("theta",), help="exp(−iθX/2)"),
    GateSpec("RY", "Rotation about Y", params=("theta",), help="exp(−iθY/2)"),
    GateSpec("RZ", "Rotation about Z", params=("theta",), help="exp(−iθZ/2)"),
    GateSpec("P", "Phase", params=("phi",), help="Phase of φ on |1⟩"),
    GateSpec("U", "General rotation", params=("theta", "phi", "lambda"), help="Any one-qubit gate"),
    GateSpec("CX", "CNOT", controls=1, base="X", help="Flips the target when the control is |1⟩"),
    GateSpec("CY", "Controlled Y", controls=1, base="Y"),
    GateSpec("CZ", "Controlled Z", controls=1, base="Z", help="Phase flip when both are |1⟩"),
    GateSpec("CH", "Controlled Hadamard", controls=1, base="H"),
    GateSpec("CP", "Controlled phase", controls=1, base="P", params=("phi",),
             help="Phase φ when both are |1⟩ (the QFT's rotations)"),
    GateSpec("CRZ", "Controlled RZ", controls=1, base="RZ", params=("theta",)),
    GateSpec("SWAP", "Swap", targets=2, help="Exchanges two qubits"),
    GateSpec("CCX", "Toffoli", controls=2, base="X", help="NOT when both controls are |1⟩"),
    GateSpec("CSWAP", "Fredkin", controls=1, targets=2, base="SWAP", help="Swap when the control is |1⟩"),
)}


def gate(name: str) -> GateSpec:
    try:
        return GATES[name]
    except KeyError:
        raise KeyError(f"unknown gate {name!r}; known: {', '.join(GATES)}") from None
