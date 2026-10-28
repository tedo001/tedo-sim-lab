"""Quantum ML: the batched simulator against the Simulation Lab's, parameter-shift gradients against
finite differences, the fidelity kernel, and the runner end to end with predictions afterwards."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from core.common.cancel import CancelToken  # noqa: E402
from core.common.paths import AppPaths  # noqa: E402
from core.experiment_engine.spec import dump_spec  # noqa: E402
from core.experiment_engine.worker import run_evaluation, run_experiment  # noqa: E402
from labs.quantum_ml.kernel import fidelity_kernel, kernel_alignment  # noqa: E402
from labs.quantum_ml.presets import QUANTUM_PRESETS, quantum_spec  # noqa: E402
from labs.quantum_ml.runner import QuantumRunner, build_kernel_svm, build_vqc  # noqa: E402
from labs.quantum_ml.sim import encode, expectation_z  # noqa: E402
from labs.quantum_ml.vqc import VariationalClassifier  # noqa: E402
from tests.labs.test_classical_ml import Recorder  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def test_encoding_matches_the_simulation_lab() -> None:
    from labs.simulation.quantum.circuit import Circuit, Op
    from labs.simulation.quantum.statevector import simulate

    x = np.array([[0.3, 1.7, 2.9]])
    psi = encode(x, "angle")
    circuit = Circuit(3, [Op("RY", (q,), params=(float(x[0, q]),), column=0) for q in range(3)])
    result = simulate(circuit)
    assert np.allclose(expectation_z(psi, 3)[0], [result.bloch(q)[2] for q in range(3)])
    assert np.allclose(expectation_z(psi, 3)[0], np.cos(x[0]))  # RY(θ)|0⟩ has ⟨Z⟩ = cos θ
    zz = encode(x, "zz")
    assert np.isclose(np.sum(np.abs(zz) ** 2), 1.0)


def test_parameter_shift_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    model = VariationalClassifier(3, 3, layers=2, encoding="zz", seed=3)
    x, y = rng.uniform(0, np.pi, (6, 3)), rng.integers(0, 3, 6)
    _loss, grad = model.gradient(x, y)
    numeric = np.empty_like(grad)
    for k in range(model.n_params):
        step = np.zeros(model.n_params)
        step[k] = 1e-6
        numeric[k] = (model.loss(x, y, model.params + step) - model.loss(x, y, model.params - step)) / 2e-6
    assert np.allclose(grad, numeric, atol=1e-6)
    with pytest.raises(ValueError, match="classes need"):
        VariationalClassifier(2, 3)


def test_fidelity_kernel_is_a_kernel() -> None:
    x = np.random.default_rng(0).uniform(0, np.pi, (8, 2))
    k = fidelity_kernel(x)
    assert np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T)
    assert np.linalg.eigvalsh(k).min() > -1e-9
    assert -1 <= kernel_alignment(k, np.array([0, 1] * 4)) <= 1
    assert build_vqc(classes=2, qubits=2).n_params == 8 and build_kernel_svm().kernel == "precomputed"


def test_presets_and_validation(workspace) -> None:
    runner = QuantumRunner()
    for preset in QUANTUM_PRESETS:
        assert runner.validate(preset.make()) == [], preset.id
    assert "unknown setting" in " ".join(runner.validate(quantum_spec("x", "two_moons", "vqc", depth=3)))
    assert "between 2 and 10" in " ".join(runner.validate(quantum_spec("x", "two_moons", "vqc", qubits=12)))


def test_vqc_and_kernel_runs_end_to_end(workspace) -> None:
    from labs.inference.runs import predict, sweep

    for name, spec in (("vqc", quantum_spec("moons", "two_moons", "vqc", epochs=4, qubits=2, layers=2)),
                       ("kernel", quantum_spec("circles", "circles", "quantum_kernel_svm", qubits=2,
                                               subset=120))):
        run_dir = workspace.experiments / name
        dump_spec(spec, run_dir / "experiment.yaml")
        result = run_experiment(run_dir, Recorder(), CancelToken())
        assert result.status == "completed", result.error
        results = json.loads((run_dir / "results.json").read_text())
        assert results["boundary"] and results["classes"] and result.metrics["test_acc"] > 0.5
        assert ("circuit" in results) == (name == "vqc") and ("kernel" in results) == (name == "kernel")
        assert (run_dir / "checkpoints" / "best.npz").is_file()
        again = run_evaluation(run_dir, Recorder(), CancelToken(),
                               checkpoint=run_dir / "checkpoints" / "best.npz",
                               split="test")
        assert again.metrics["test_acc"] == pytest.approx(result.metrics["test_acc"])
        answer = predict({"run": str(run_dir), "inputs": [{"x1": 0.0, "x2": 0.5}]}, lambda _: None,
                         lambda: False, workspace.workspace)
        assert answer["model"]["kind"] == "quantum" and len(answer["predictions"][0]["classes"]) == 2
        swept = sweep({"run": str(run_dir), "base": {"x1": 0.0, "x2": 0.0}, "feature": "x1", "low": -1,
                       "high": 1, "steps": 5}, lambda _: None, lambda: False, workspace.workspace)
        assert len(swept["values"]) == 5 and np.allclose(np.sum(swept["probabilities"], axis=0), 1.0)
