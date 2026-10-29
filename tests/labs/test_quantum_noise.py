"""Noisy quantum simulation (density matrices, depolarizing, T1/T2, readout, shots) and the quantum
evaluation protocol: classical baselines on the same split, resources, a noise comparison, and a
report that says simulator, not hardware, and claims no quantum advantage (the acceptance test)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from core.common.cancel import CancelToken  # noqa: E402
from core.common.paths import AppPaths  # noqa: E402
from core.experiment_engine.spec import QuantumTrainingSpec, SimulatorSpec, dump_spec  # noqa: E402
from core.experiment_engine.worker import run_experiment  # noqa: E402
from labs.quantum_ml.presets import quantum_spec  # noqa: E402
from labs.simulation.quantum.circuit import PRESETS  # noqa: E402
from labs.simulation.quantum.noise import (  # noqa: E402
    REFERENCE,
    NoiseModel,
    apply_kraus,
    depolarizing_kraus,
    readout_probabilities,
    relaxation_kraus,
    simulate_noisy,
)
from labs.simulation.quantum.statevector import simulate  # noqa: E402
from tests.labs.test_classical_ml import Recorder  # noqa: E402


def test_ideal_density_matrices_equal_the_state_vector():
    for _name, (_text, circuit) in PRESETS.items():
        psi = simulate(circuit).state
        assert np.allclose(simulate_noisy(circuit, NoiseModel()), np.outer(psi, psi.conj()), atol=1e-12)
        noisy = simulate_noisy(circuit, REFERENCE)
        assert np.isclose(np.trace(noisy).real, 1.0) and np.linalg.eigvalsh(noisy).min() > -1e-12
        assert np.trace(noisy @ noisy).real < 1.0  # noise mixes the state


def test_channels():
    zero = np.array([[[1, 0], [0, 0]]], dtype=complex)
    assert np.allclose(apply_kraus(zero, depolarizing_kraus(1.0, 1), (0,), 1), np.eye(2) / 2)
    bell = simulate_noisy(PRESETS["Bell state"][1], NoiseModel())[None]
    assert np.allclose(apply_kraus(bell, depolarizing_kraus(1.0, 2), (0, 1), 2), np.eye(4) / 4)
    one = np.array([[[0, 0], [0, 1]]], dtype=complex)
    for channel in relaxation_kraus(1000, 10, 10):
        one = apply_kraus(one, channel, (0,), 1)
    assert one[0, 1, 1].real == pytest.approx(np.exp(-0.1))
    flipped = readout_probabilities(np.array([1.0, 0, 0, 0]), 2, 0.1)
    assert flipped == pytest.approx([0.81, 0.09, 0.09, 0.01])
    with pytest.raises(ValueError, match="2·T1"):
        NoiseModel(t1_us=10, t2_us=30)
    with pytest.raises(ValueError):
        SimulatorSpec(t1_us=10, t2_us=30)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def _run(paths, spec, name):
    run_dir = paths.experiments / name / "run"
    dump_spec(spec, run_dir / "experiment.yaml")
    result = run_experiment(run_dir, Recorder(), CancelToken())
    assert result.status == "completed", result.error
    return run_dir, json.loads((run_dir / "results.json").read_text())


@pytest.mark.parametrize("model", ["vqc", "quantum_kernel_svm"])
def test_noisy_run_records_the_protocol_and_its_report_claims_nothing(workspace, model):
    extra = {"layers": 1} if model == "vqc" else {}
    base = quantum_spec("noisy", "two_moons", model, epochs=2, qubits=2, **extra)
    training = QuantumTrainingSpec(epochs=2, simulator=SimulatorSpec(mode="noisy", shots=500),
                                   baselines=("logistic_regression", "svm_rbf", "mlp"))
    data = base.data.model_copy(update={"subset": 80})
    spec = base.model_copy(update={"training": training, "data": data})
    run_dir, results = _run(workspace, spec, model)
    protocol = results["protocol"]
    assert protocol["simulator"]["mode"] == "noisy" and protocol["simulator"]["shots"] == 500
    assert protocol["simulator"]["noise"]["depolarizing_2q"] == 0.01
    assert {b["model"] for b in protocol["baselines"]} == {"logistic_regression", "svm_rbf", "mlp"}
    assert protocol["split"]["same_split_for_baselines"]
    assert protocol["split"]["test"] == int(np.sum(results["matrix"]))
    resources = protocol["resources"]
    assert resources["qubits"] == 2 and resources["gates_1q"] > 0 and resources["circuit_evaluations"] > 0
    assert resources["shots"] > 0 and resources["state_memory_bytes"] == 16 * 4 ** 2
    assert [row["factor"] for row in protocol["noise_comparison"]["rows"]] == [0.0, 0.5, 1.0, 2.0]
    assert protocol["claims"]["hardware"] is False
    report = (run_dir / "reproduction" / "report.md").read_text()
    assert "Simulator results, not hardware results" in report
    assert "quantum advantage not claimed" in report
    assert "not evidence of quantum advantage" in report  # the observed limitation, too
    assert "advantage is" not in report.lower() and "demonstrates quantum advantage" not in report.lower()
    assert "Classical baselines on the same split" in report


def test_ideal_run_compares_against_the_reference_device(workspace):
    spec = quantum_spec("ideal", "two_moons", "vqc", epochs=2, qubits=2, layers=1, subset=60)
    _run_dir, results = _run(workspace, spec, "ideal")
    protocol = results["protocol"]
    assert protocol["simulator"]["kind"] == "state vector"
    assert "reference device" in protocol["noise_comparison"]["reference"]
    assert protocol["noise_comparison"]["rows"][0]["noise"]["depolarizing_1q"] == 0.0
