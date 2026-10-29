"""The experiment specification: everything that decides what a run does.

Serialised as ``experiment.yaml``. Validation is strict (unknown keys are
errors, so a typo cannot silently fall back to a default), and the dump is
canonical: loading a dumped file and dumping it again gives identical bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.common.vocab import TABULAR_TASKS, Task

__all__ = ["CheckpointSpec", "DataSpec", "EarlyStoppingSpec", "ExperimentSpec", "ModelSpec",
           "OptimizerSpec", "QuantumTrainingSpec", "ResearchSpec", "RuntimeSpec", "SchedulerSpec",
           "SearchSpec", "SimulatorSpec", "SklearnTrainingSpec", "SpecError", "SplitSpec", "TRAINING_KINDS",
           "TorchTrainingSpec", "TransformSpec", "dump_spec", "dump_spec_text", "load_spec",
           "load_spec_text", "spec_hash"]

SCHEMA_VERSION = 1


class SpecError(ValueError):
    """An experiment.yaml that cannot be used; the message lists every problem."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TransformSpec(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class SplitSpec(_Strict):
    #: Fraction of the training split held out for validation.
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    stratify: bool = True


class DataSpec(_Strict):
    dataset: str
    split: SplitSpec = SplitSpec()
    preprocessing: tuple[TransformSpec, ...] = ()
    augmentation: tuple[TransformSpec, ...] = ()
    #: ``(height, width)``; ``None`` = the dataset's native size.
    input_size: tuple[int, int] | None = None
    #: Cap on samples per split, for quick runs. Recorded like everything else.
    subset: int | None = Field(default=None, gt=0)
    #: Dataset checks before training (duplicates, split leakage, label conflicts, imbalance, licence):
    #: ``warn`` records and reports them, ``fail`` stops the run on any error, ``off`` skips them.
    validation: Literal["off", "warn", "fail"] = "warn"


class ModelSpec(_Strict):
    model: str
    #: A ``WeightsInfo.id`` from the model card; ``None`` = train from scratch.
    pretrained: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class OptimizerSpec(_Strict):
    name: Literal["sgd", "adam", "adamw"] = "adam"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.9, ge=0, lt=1)


class SchedulerSpec(_Strict):
    name: Literal["none", "step", "cosine", "plateau", "onecycle"] = "none"
    params: dict[str, Any] = Field(default_factory=dict)


class CheckpointSpec(_Strict):
    strategy: Literal["best", "last", "best_and_last", "every_n"] = "best_and_last"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    every_n: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _every_n(self) -> CheckpointSpec:
        if (self.strategy == "every_n") != (self.every_n is not None):
            raise ValueError("every_n is required with strategy 'every_n' and only then")
        return self


class EarlyStoppingSpec(_Strict):
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=5, gt=0)
    min_delta: float = Field(default=0.0, ge=0)


class TorchTrainingSpec(_Strict):
    kind: Literal["torch"] = "torch"
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=64, gt=0)
    optimizer: OptimizerSpec = OptimizerSpec()
    scheduler: SchedulerSpec = SchedulerSpec()
    loss: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    checkpoint: CheckpointSpec = CheckpointSpec()
    early_stopping: EarlyStoppingSpec | None = None
    grad_clip_norm: float | None = Field(default=None, gt=0)
    #: Cap on batches per epoch, for quick runs. Recorded like everything else.
    max_steps_per_epoch: int | None = Field(default=None, gt=0)


class SearchSpec(_Strict):
    method: Literal["grid", "random"] = "grid"
    space: dict[str, tuple[Any, ...]]
    n_iter: int | None = Field(default=None, gt=0)
    scoring: str | None = None


class SklearnTrainingSpec(_Strict):
    kind: Literal["sklearn"] = "sklearn"
    #: For CSV datasets: the target column and, optionally, the feature columns to use.
    target: str | None = None
    features: tuple[str, ...] | None = None
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    cv_folds: int | None = Field(default=None, ge=2)
    search: SearchSpec | None = None


class SimulatorSpec(_Strict):
    """Where a quantum model runs: an exact (``ideal``) state-vector simulation, or a ``noisy``
    density-matrix simulation of an imperfect device (depolarizing errors after one- and two-qubit
    gates, T1/T2 relaxation for each gate's duration, readout bit flips); ``shots`` estimates each
    expectation from that many measurements instead of reading it exactly. Never hardware."""
    mode: Literal["ideal", "noisy"] = "ideal"
    depolarizing_1q: float = Field(default=0.001, ge=0, le=1)
    depolarizing_2q: float = Field(default=0.01, ge=0, le=1)
    #: Relaxation (T1) and coherence (T2) times in microseconds; ``None`` = no relaxation.
    t1_us: float | None = Field(default=100.0, gt=0)
    t2_us: float | None = Field(default=80.0, gt=0)
    gate_time_1q_ns: float = Field(default=50.0, gt=0)
    gate_time_2q_ns: float = Field(default=300.0, gt=0)
    readout_error: float = Field(default=0.02, ge=0, le=0.5)
    shots: int | None = Field(default=None, gt=0, le=1_000_000)

    @model_validator(mode="after")
    def _times(self) -> SimulatorSpec:
        if (self.t1_us is None) != (self.t2_us is None):
            raise ValueError("give t1_us and t2_us together (or neither)")
        if self.t1_us is not None and self.t2_us > 2 * self.t1_us:
            raise ValueError("t2_us can be at most 2 × t1_us")
        return self


class QuantumTrainingSpec(_Strict):
    """Quantum classifiers: a variational circuit trained for ``epochs`` with Adam (``lr``, batches
    of ``batch_size``), or a quantum-kernel SVM fitted once (which ignores the three)."""
    kind: Literal["quantum"] = "quantum"
    epochs: int = Field(default=30, gt=0, le=1000)
    batch_size: int = Field(default=16, gt=0)
    lr: float = Field(default=0.1, gt=0)
    test_fraction: float = Field(default=0.25, gt=0, lt=1)
    simulator: SimulatorSpec = SimulatorSpec()
    #: Classical models fitted on the same split and features, for comparison.
    baselines: tuple[Literal["logistic_regression", "svm_rbf", "mlp"], ...] = ("logistic_regression",
                                                                             "svm_rbf")


class RuntimeSpec(_Strict):
    #: ``auto`` picks CUDA, then MPS, then CPU.
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(:\d+)?|mps)$")
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    num_workers: int = Field(default=0, ge=0)
    deterministic: bool = False
    #: A remote target from the workspace's ``configs/remotes.yaml`` to run on (``None`` = here).
    #: Where a run happens is not what it computes, so it is left out of :func:`spec_hash`.
    target: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")


TrainingSpec = Annotated[TorchTrainingSpec | SklearnTrainingSpec | QuantumTrainingSpec,
                         Field(discriminator="kind")]

#: The training kinds each task can use; every other task trains with ``torch``.
TRAINING_KINDS: dict[Task, frozenset[str]] = {
    **{task: frozenset({"sklearn"}) for task in TABULAR_TASKS},
    Task.TEXT_CLASSIFICATION: frozenset({"sklearn", "torch"}),
    Task.QUANTUM_CLASSIFICATION: frozenset({"quantum"}),
}


class ResearchSpec(_Strict):
    """Why the experiment is run, for its reproducibility report; never changes a result."""
    question: str = ""
    hypothesis: str = ""
    #: The run this one is measured against (a run id), or ``None``.
    baseline: str | None = None
    limitations: str = ""


class ExperimentSpec(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    description: str = ""
    tags: tuple[str, ...] = ()
    research: ResearchSpec = ResearchSpec()
    task: Task
    seed: int = 42
    data: DataSpec
    model: ModelSpec
    training: TrainingSpec
    runtime: RuntimeSpec = RuntimeSpec()

    @model_validator(mode="after")
    def _training_fits_task(self) -> ExperimentSpec:
        wanted = TRAINING_KINDS.get(self.task, frozenset({"torch"}))
        if self.training.kind not in wanted:
            kinds = " or ".join(repr(kind) for kind in sorted(wanted))
            raise ValueError(f"task {self.task.value!r} trains with training.kind {kinds}, not "
                             f"{self.training.kind!r}")
        return self


def _problems(error: ValidationError) -> str:
    return "\n".join(f"  {'.'.join(map(str, e['loc'])) or 'spec'}: {e['msg']}" for e in error.errors())


def load_spec_text(text: str) -> ExperimentSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise SpecError("an experiment spec is a mapping of section: values")
    try:
        return ExperimentSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"invalid experiment spec:\n{_problems(exc)}") from None


def load_spec(path: Path) -> ExperimentSpec:
    return load_spec_text(path.read_text(encoding="utf-8"))


def dump_spec_text(spec: ExperimentSpec) -> str:
    """Canonical YAML: every field, in declaration order, defaults included."""
    data = spec.model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False,
                          width=100)


def dump_spec(spec: ExperimentSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_spec_text(spec), encoding="utf-8")
    return path


def spec_hash(spec: ExperimentSpec) -> str:
    """SHA-256 of what decides the result: everything except name, description, tags, the research
    notes, whether the data checks stop a run and which machine runs it."""
    data = spec.model_dump(mode="json", exclude={"name": True, "description": True, "tags": True,
                                                 "research": True, "data": {"validation"},
                                                 "runtime": {"target"}})
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
