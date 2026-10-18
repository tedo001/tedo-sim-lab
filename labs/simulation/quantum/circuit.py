"""A quantum circuit: qubits and the gates placed on them, column by column (time runs left to
right). Circuits are plain data — saved as JSON, exported as OpenQASM 2 — and simulated by
:mod:`labs.simulation.quantum.statevector`. Presets are the textbook circuits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gates import gate

__all__ = ["MAX_QUBITS", "PRESETS", "Circuit", "Op", "preset"]

MAX_QUBITS = 10

_QASM = {"X": "x", "Y": "y", "Z": "z", "H": "h", "S": "s", "Sdg": "sdg", "T": "t", "Tdg": "tdg",
         "SX": "sx", "RX": "rx", "RY": "ry", "RZ": "rz", "P": "u1", "U": "u3", "CX": "cx", "CY": "cy",
         "CZ": "cz", "CH": "ch", "CP": "cu1", "CRZ": "crz", "SWAP": "swap", "CCX": "ccx", "CSWAP": "cswap"}


@dataclass(frozen=True)
class Op:
    gate: str
    #: Qubits the gate acts on (two for SWAP).
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    #: The time step it is drawn in (ops in one column act on different qubits).
    column: int = 0

    @property
    def qubits(self) -> tuple[int, ...]:
        return (*self.controls, *self.targets)

    def label(self) -> str:
        if not self.params:
            return self.gate
        return f"{self.gate}({', '.join(_angle(p) for p in self.params)})"


@dataclass
class Circuit:
    qubits: int = 2
    ops: list[Op] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.qubits <= MAX_QUBITS:
            raise ValueError(f"a circuit has 1 to {MAX_QUBITS} qubits")
        for op in self.ops:
            self.check(op)

    def check(self, op: Op) -> None:
        """``ValueError`` when ``op`` does not fit this circuit."""
        spec = gate(op.gate)
        if len(op.targets) != spec.targets or len(op.controls) != spec.controls:
            raise ValueError(f"{op.gate} needs {spec.targets} target(s) and {spec.controls} control(s)")
        if len(op.params) != len(spec.params):
            raise ValueError(f"{op.gate} takes {len(spec.params)} parameter(s)")
        qubits = op.qubits
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"{op.gate} uses a qubit twice")
        if any(not 0 <= q < self.qubits for q in qubits):
            raise ValueError(f"{op.gate} uses a qubit outside 0…{self.qubits - 1}")
        if op.column < 0:
            raise ValueError("columns start at 0")

    def ordered(self) -> list[Op]:
        """Ops in the order they act: by column, then as placed."""
        return [op for _, op in sorted(enumerate(self.ops), key=lambda item: (item[1].column, item[0]))]

    @property
    def columns(self) -> int:
        return max((op.column for op in self.ops), default=-1) + 1

    def occupied(self, column: int) -> set[int]:
        """Qubits a gate in ``column`` already uses (including the wires between control and target)."""
        used: set[int] = set()
        for op in self.ops:
            if op.column == column:
                low, high = min(op.qubits), max(op.qubits)
                used.update(range(low, high + 1))
        return used

    def add(self, op: Op) -> Op:
        """Place ``op``; if its column is taken on those qubits, it goes in the next free one."""
        self.check(op)
        span = set(range(min(op.qubits), max(op.qubits) + 1))
        column = op.column
        while self.occupied(column) & span:
            column += 1
        placed = Op(op.gate, op.targets, op.controls, op.params, column)
        self.ops.append(placed)
        return placed

    def remove_at(self, column: int, qubit: int) -> Op | None:
        """Remove the gate drawn at (``column``, ``qubit``), if any."""
        for op in reversed(self.ops):
            if op.column == column and min(op.qubits) <= qubit <= max(op.qubits):
                self.ops.remove(op)
                return op
        return None

    def resized(self, qubits: int) -> Circuit:
        """The same circuit on ``qubits`` qubits; gates on removed qubits are dropped."""
        kept = [op for op in self.ops if max(op.qubits) < qubits]
        return Circuit(qubits, kept, self.name)

    # Files -----------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"qubits": self.qubits, "name": self.name,
                "ops": [{"gate": op.gate, "targets": list(op.targets), "controls": list(op.controls),
                         "params": list(op.params), "column": op.column} for op in self.ops]}

    @classmethod
    def from_dict(cls, data: dict) -> Circuit:
        ops = [Op(item["gate"], tuple(item["targets"]), tuple(item.get("controls", ())),
                  tuple(float(p) for p in item.get("params", ())), int(item.get("column", 0)))
               for item in data.get("ops", [])]
        return cls(int(data["qubits"]), ops, data.get("name", ""))

    def to_qasm(self) -> str:
        """OpenQASM 2.0 (qelib1.inc), measuring every qubit at the end."""
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.qubits}];",
                 f"creg c[{self.qubits}];"]
        for op in self.ordered():
            name = _QASM[op.gate]
            args = f"({','.join(repr(float(p)) for p in op.params)})" if op.params else ""
            qubits = ",".join(f"q[{q}]" for q in op.qubits)
            lines.append(f"{name}{args} {qubits};")
        lines.append("measure q -> c;")
        return "\n".join(lines) + "\n"


def _angle(value: float) -> str:
    """π fractions where they are exact (π/2, 3π/4), else a decimal."""
    for denominator in (1, 2, 3, 4, 6, 8, 16):
        numerator = value * denominator / math.pi
        if abs(numerator - round(numerator)) < 1e-9 and round(numerator) != 0:
            top = round(numerator)
            sign = "−" if top < 0 else ""
            top = abs(top)
            head = "π" if top == 1 else f"{top}π"
            return sign + (head if denominator == 1 else f"{head}/{denominator}")
    return f"{value:.3g}"


def _ops(*items: tuple) -> list[Op]:
    return [Op(*item) for item in items]


#: name → (description, circuit)
PRESETS: dict[str, tuple[str, Circuit]] = {
    "Superposition": ("A Hadamard puts one qubit in an equal superposition of |0⟩ and |1⟩.",
                      Circuit(1, _ops(("H", (0,), (), (), 0)))),
    "Bell state": ("Hadamard then CNOT: two qubits entangled, measured 00 or 11, never mixed.",
                   Circuit(2, _ops(("H", (0,), (), (), 0), ("CX", (1,), (0,), (), 1)))),
    "GHZ state": ("Three qubits in (|000⟩ + |111⟩)/√2: all agree, none has a value of its own.",
                  Circuit(3, _ops(("H", (0,), (), (), 0), ("CX", (1,), (0,), (), 1),
                                  ("CX", (2,), (1,), (), 2)))),
    "Teleportation": ("Qubit 0's state (RY 1.1) moves to qubit 2 through a Bell pair; the "
                      "measurements are deferred, so the corrections are controlled gates.",
                      Circuit(3, _ops(("RY", (0,), (), (1.1,), 0), ("H", (1,), (), (), 0),
                                      ("CX", (2,), (1,), (), 1), ("CX", (1,), (0,), (), 2),
                                      ("H", (0,), (), (), 3), ("CX", (2,), (1,), (), 4),
                                      ("CZ", (2,), (0,), (), 5)))),
    "Deutsch–Jozsa": ("One oracle call decides that f(x) = x0 ⊕ x1 is balanced: the inputs never "
                      "read 00.",
                      Circuit(3, _ops(("X", (2,), (), (), 0), ("H", (0,), (), (), 1), ("H", (1,), (), (), 1),
                                      ("H", (2,), (), (), 1), ("CX", (2,), (0,), (), 2),
                                      ("CX", (2,), (1,), (), 3), ("H", (0,), (), (), 4),
                                      ("H", (1,), (), (), 4)))),
    "Grover search": ("Two qubits, the oracle marks |11⟩; one Grover iteration finds it with "
                      "certainty.",
                      Circuit(2, _ops(("H", (0,), (), (), 0), ("H", (1,), (), (), 0),
                                      ("CZ", (1,), (0,), (), 1), ("H", (0,), (), (), 2),
                                      ("H", (1,), (), (), 2), ("X", (0,), (), (), 3),
                                      ("X", (1,), (), (), 3), ("CZ", (1,), (0,), (), 4),
                                      ("X", (0,), (), (), 5), ("X", (1,), (), (), 5),
                                      ("H", (0,), (), (), 6), ("H", (1,), (), (), 6)))),
    "Quantum Fourier transform": ("The QFT of |001⟩: every outcome equally likely, with phases "
                                  "rising in steps of π/4.",
                                  Circuit(3, _ops(("X", (2,), (), (), 0), ("H", (0,), (), (), 1),
                                                  ("CP", (0,), (1,), (math.pi / 2,), 2),
                                                  ("CP", (0,), (2,), (math.pi / 4,), 3),
                                                  ("H", (1,), (), (), 4),
                                                  ("CP", (1,), (2,), (math.pi / 2,), 5),
                                                  ("H", (2,), (), (), 6),
                                                  ("SWAP", (0, 2), (), (), 7)))),
}


def preset(name: str) -> Circuit:
    description, circuit = PRESETS[name]
    return Circuit.from_dict({**circuit.to_dict(), "name": name})
