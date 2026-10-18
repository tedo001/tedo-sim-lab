"""Quantum engines: the circuit simulator against an independent reference, the textbook
circuits, and the Schrödinger solver against exact results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from labs.simulation.quantum import GATES, PRESETS, Circuit, Op, expectation, preset, purity, sample, simulate
from labs.simulation.quantum.statevector import unitary_of
from labs.simulation.quantum.waves import WaveSetup, WaveSimulation, eigenstates


def reference(circuit: Circuit) -> np.ndarray:
    """A deliberately different simulator: every op as a full 2ⁿ matrix built bit by bit."""
    n = circuit.qubits
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    for op in circuit.ordered():
        g = GATES[op.gate].matrix(*op.params)
        full = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for column in range(2 ** n):
            bits = [(column >> (n - 1 - q)) & 1 for q in range(n)]
            if not all(bits[c] for c in op.controls):
                full[column, column] = 1
                continue
            sub = int("".join(str(bits[t]) for t in op.targets), 2)
            for out in range(g.shape[0]):
                new = list(bits)
                for i, t in enumerate(op.targets):
                    new[t] = (out >> (len(op.targets) - 1 - i)) & 1
                full[int("".join(map(str, new)), 2), column] += g[out, sub]
        state = full @ state
    return state


def random_circuit(rng: np.random.Generator, qubits: int = 3, length: int = 14) -> Circuit:
    circuit = Circuit(qubits)
    names = list(GATES)
    for _ in range(length):
        spec = GATES[rng.choice(names)]
        if spec.targets + spec.controls > qubits:
            continue
        chosen = [int(q) for q in rng.permutation(qubits)[: spec.targets + spec.controls]]
        params = tuple(float(p) for p in rng.uniform(-math.pi, math.pi, len(spec.params)))
        circuit.add(Op(spec.name, tuple(chosen[spec.controls:]), tuple(chosen[: spec.controls]), params))
    return circuit


@pytest.mark.parametrize("seed", range(6))
def test_the_simulator_matches_an_independent_reference(seed: int) -> None:
    circuit = random_circuit(np.random.default_rng(seed), qubits=3 + seed % 2)
    assert np.allclose(simulate(circuit).state, reference(circuit), atol=1e-10)


def test_every_gate_is_unitary() -> None:
    rng = np.random.default_rng(1)
    for spec in GATES.values():
        m = spec.matrix(*rng.uniform(-3, 3, len(spec.params)))
        assert np.allclose(m.conj().T @ m, np.eye(len(m))), spec.name


def test_textbook_circuits() -> None:
    bell = simulate(preset("Bell state"))
    assert np.allclose(bell.probabilities, [0.5, 0, 0, 0.5])
    assert expectation(bell.state, "ZZ") == pytest.approx(1)
    assert expectation(bell.state, "XX") == pytest.approx(1)
    assert expectation(bell.state, "ZI") == pytest.approx(0, abs=1e-12)
    assert purity(bell.state, 0, 2) == pytest.approx(0.5) and np.allclose(bell.bloch(0), 0)
    ghz = simulate(preset("GHZ state")).probabilities
    assert ghz[0] == pytest.approx(0.5) and ghz[-1] == pytest.approx(0.5)
    assert simulate(preset("Grover search")).probabilities[3] == pytest.approx(1)
    dj = simulate(preset("Deutsch–Jozsa")).probabilities.reshape(2, 2, 2)
    assert dj[0, 0].sum() == pytest.approx(0, abs=1e-12)  # the inputs never read 00: balanced
    teleported = simulate(preset("Teleportation")).bloch(2)
    assert np.allclose(teleported, (math.sin(1.1), 0, math.cos(1.1)), atol=1e-10)
    assert set(PRESETS) >= {"Bell state", "Teleportation", "Quantum Fourier transform"}


def test_the_qft_circuit_is_the_discrete_fourier_transform() -> None:
    circuit = Circuit(3, [op for op in preset("Quantum Fourier transform").ops if op.column > 0])
    omega = np.exp(2j * np.pi / 8)
    dft = np.array([[omega ** (j * k) for j in range(8)] for k in range(8)]) / math.sqrt(8)
    assert np.allclose(unitary_of(circuit), dft)


def test_sampling_is_reproducible_and_honest() -> None:
    state = simulate(preset("Teleportation")).state
    first, again = sample(state, 4000, seed=3), sample(state, 4000, seed=3)
    assert first == again and sum(first.values()) == 4000
    probs = simulate(preset("Teleportation")).probabilities
    for label, count in first.items():
        assert abs(count / 4000 - probs[int(label, 2)]) < 0.03


def test_circuit_editing_files_and_qasm() -> None:
    circuit = Circuit(3)
    first = circuit.add(Op("H", (0,)))
    second = circuit.add(Op("CX", (2,), (0,)))  # column 0 is taken on qubit 0: next column
    assert (first.column, second.column) == (0, 1) and circuit.occupied(1) == {0, 1, 2}
    third = circuit.add(Op("X", (1,), column=1))
    assert third.column == 2
    assert circuit.remove_at(1, 1) == second and len(circuit.ops) == 2
    for bad in (Op("CX", (0,), (0,)), Op("RX", (0,)), Op("H", (5,)), Op("SWAP", (1,))):
        with pytest.raises(ValueError):
            circuit.check(bad)
    again = Circuit.from_dict(preset("Quantum Fourier transform").to_dict())
    assert np.allclose(simulate(again).state, simulate(preset("Quantum Fourier transform")).state)
    qasm = preset("Quantum Fourier transform").to_qasm()
    assert qasm.startswith("OPENQASM 2.0;") and "cu1(1.5707963267948966) q[1],q[0];" in qasm
    assert "swap q[0],q[2];" in qasm and qasm.rstrip().endswith("measure q -> c;")
    assert Op("RZ", (0,), params=(math.pi / 2,)).label() == "RZ(π/2)"
    assert circuit.resized(1).ops == [first]
    with pytest.raises(ValueError):
        Circuit(11)


def test_harmonic_oscillator_and_box_energies() -> None:
    energies, _x, states = eigenstates(WaveSetup(potential="harmonic", omega=0.5, length=60, points=2048), 5)
    assert np.allclose(energies, [0.25, 0.75, 1.25, 1.75, 2.25], atol=1e-3)
    assert np.sum(states[:, 0] ** 2) * (_x[1] - _x[0]) == pytest.approx(1, rel=1e-6)
    box, _, _ = eigenstates(WaveSetup(potential="box", width=10, length=20, points=2048), 3)
    exact = np.array([1, 4, 9]) * math.pi ** 2 / (2 * 10 ** 2)
    assert np.allclose(box, exact, rtol=0.01)


def test_tunnelling_matches_the_plane_wave_formula() -> None:
    setup = WaveSetup(potential="barrier", height=1.0, width=1.0, x0=-40, sigma=10.0, k0=1.2,
                      length=300, points=8192, dt=0.02, absorbing=False)
    simulation = WaveSimulation(setup)
    simulation.step(int(55 / setup.dt))
    energy = 0.5 * setup.k0 ** 2
    kappa = math.sqrt(2 * (setup.height - energy))
    sinh2 = math.sinh(kappa * setup.width) ** 2
    exact = 1 / (1 + setup.height ** 2 * sinh2 / (4 * energy * (setup.height - energy)))
    transmitted, reflected = simulation.beyond(0.5), simulation.before(-0.5)
    assert simulation.norm() == pytest.approx(1, abs=1e-9)
    assert transmitted == pytest.approx(exact, abs=0.01)
    assert transmitted + reflected == pytest.approx(1, abs=5e-3)  # the packet's tail is still in the barrier


def test_free_motion_conserves_energy_and_edges_absorb() -> None:
    free = WaveSimulation(WaveSetup(potential="free", absorbing=False, length=200, points=4096))
    start, mean = free.energy(), free.mean_x()
    free.step(300)
    assert free.energy() == pytest.approx(start, rel=1e-9) and start == pytest.approx(0.5 * 1.2 ** 2 + 1 / 32)
    assert free.mean_x() == pytest.approx(mean + 1.2 * free.time, abs=0.05)  # <x> moves at k0
    leaving = WaveSimulation(WaveSetup(potential="free", x0=20, k0=3.0, length=80, points=1024))
    leaving.step(2000)
    assert leaving.norm() < 0.05
    with pytest.raises(ValueError):
        WaveSetup(points=1000).check()
    assert WaveSetup.from_dict({"height": 2.0, "unknown": 1}).height == 2.0
