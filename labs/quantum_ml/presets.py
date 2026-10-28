"""Ready-made quantum experiments for the Quantum ML lab; each opens in the lab's form, where every
setting can still be changed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.common.vocab import Task
from core.experiment_engine.spec import DataSpec, ExperimentSpec, ModelSpec, QuantumTrainingSpec, RuntimeSpec

__all__ = ["QUANTUM_PRESETS", "QuantumPreset", "quantum_spec"]


def quantum_spec(name: str, dataset: str, model: str, *, epochs: int = 30, lr: float = 0.1,
                 batch_size: int = 16,
                 subset: int | None = None, **params: Any) -> ExperimentSpec:
    return ExperimentSpec(name=name, task=Task.QUANTUM_CLASSIFICATION, data=DataSpec(dataset=dataset,
                                                                                     subset=subset),
                          model=ModelSpec(model=model, params=params),
                          training=QuantumTrainingSpec(epochs=epochs, lr=lr, batch_size=batch_size),
                          runtime=RuntimeSpec(device="cpu"))


@dataclass(frozen=True)
class QuantumPreset:
    id: str
    title: str
    summary: str
    make: Callable[[], ExperimentSpec]


QUANTUM_PRESETS: tuple[QuantumPreset, ...] = (
    QuantumPreset("moons_vqc", "Two moons · variational classifier",
                  "Two qubits, three layers, angle encoding: a circuit learns a curved boundary.",
                  lambda: quantum_spec("moons-vqc", "two_moons", "vqc", epochs=25, qubits=2, layers=3)),
    QuantumPreset("circles_kernel", "Circles · quantum kernel SVM",
                  "The ZZ feature map's state overlaps as an SVM kernel; one fit, no training loop.",
                  lambda: quantum_spec("circles-quantum-kernel", "circles", "quantum_kernel_svm", qubits=2)),
    QuantumPreset("xor_vqc", "XOR · ZZ feature map",
                  "Entangling encoding for a problem no linear model can solve.",
                  lambda: quantum_spec("xor-vqc", "xor", "vqc", epochs=30, qubits=2, layers=2,
                                       encoding="zz")),
    QuantumPreset("iris_vqc", "Iris · four qubits, three classes",
                  "Four measurements on four qubits; ⟨Z⟩ of qubits 0–2 score the three species.",
                  lambda: quantum_spec("iris-vqc", "iris", "vqc", epochs=30, qubits=4, layers=2)),
    QuantumPreset("cancer_kernel", "Breast cancer · kernel on 4 principal components",
                  "30 features reduced to four principal components, one per qubit.",
                  lambda: quantum_spec("breast-cancer-quantum-kernel", "breast_cancer", "quantum_kernel_svm",
                                       qubits=4, subset=300)),
)
