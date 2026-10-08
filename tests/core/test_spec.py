"""experiment.yaml: strict validation, canonical round trip, stable hash."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from core.experiment_engine.spec import (
    ExperimentSpec,
    SpecError,
    dump_spec,
    dump_spec_text,
    load_spec,
    load_spec_text,
    spec_hash,
)

TORCH = dedent("""\
    name: mnist-cnn
    task: image_classification
    tags: [baseline]
    data:
      dataset: mnist
      split: {val_fraction: 0.1}
      preprocessing:
        - {name: normalize, params: {mean: [0.1307], std: [0.3081]}}
      augmentation:
        - {name: random_rotation, params: {degrees: 10}}
      input_size: [28, 28]
    model:
      model: simple_cnn
    training:
      kind: torch
      epochs: 3
      batch_size: 64
      optimizer: {name: adamw, lr: 0.001, weight_decay: 0.01}
      scheduler: {name: cosine}
      early_stopping: {patience: 2}
    runtime:
      device: auto
      precision: fp16
""")

SKLEARN = dedent("""\
    name: iris-xgb
    task: tabular_classification
    data: {dataset: iris}
    model:
      model: xgboost
      params: {n_estimators: 200, max_depth: 4}
    training:
      kind: sklearn
      cv_folds: 5
      search:
        method: random
        space: {max_depth: [3, 4, 6], learning_rate: [0.05, 0.1]}
        n_iter: 4
""")


@pytest.mark.parametrize("text", [TORCH, SKLEARN])
def test_yaml_spec_yaml_round_trip_is_identical(text: str, tmp_path: Path) -> None:
    spec = load_spec_text(text)
    first = dump_spec_text(spec)
    assert load_spec_text(first) == spec
    assert dump_spec_text(load_spec_text(first)) == first  # byte-identical
    file = dump_spec(spec, tmp_path / "run" / "experiment.yaml")
    assert load_spec(file) == spec


def test_dump_is_complete_and_ordered() -> None:
    dumped = dump_spec_text(load_spec_text(SKLEARN))
    keys = [line.split(":")[0] for line in dumped.splitlines() if line and not line.startswith(" ")]
    assert keys == ["schema_version", "name", "description", "tags", "task", "seed", "data", "model",
                    "training", "runtime"]
    assert "precision: fp32" in dumped  # defaults are written out


def test_tuples_survive(tmp_path: Path) -> None:
    spec = load_spec_text(TORCH)
    assert spec.data.input_size == (28, 28)
    assert load_spec_text(dump_spec_text(spec)).data.input_size == (28, 28)


@pytest.mark.parametrize("change, message", [
    (("epochs: 3", "epoch: 3"), "training.torch.epoch"),           # typo is an error, not a default
    (("epochs: 3", "epochs: 0"), "greater than 0"),
    (("device: auto", "device: gpu"), "runtime.device"),
    (("precision: fp16", "precision: fp8"), "runtime.precision"),
    (("kind: torch", "kind: tensorflow"), "training"),
    (("task: image_classification", "task: tabular_classification"), "trains with training.kind"),
    (("name: mnist-cnn", "name: ''"), "name"),
])
def test_invalid_specs_list_the_problem(change: tuple[str, str], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec_text(TORCH.replace(*change))


@pytest.mark.parametrize("text", ["- a list\n", "name: [unclosed\n", ""])
def test_non_mappings_are_rejected(text: str) -> None:
    with pytest.raises(SpecError):
        load_spec_text(text)


def test_checkpoint_every_n_rules() -> None:
    with pytest.raises(SpecError, match="every_n"):
        load_spec_text(TORCH.replace("early_stopping: {patience: 2}",
                                     "checkpoint: {strategy: every_n}"))
    spec = load_spec_text(TORCH.replace("early_stopping: {patience: 2}",
                                        "checkpoint: {strategy: every_n, every_n: 2}"))
    assert spec.training.checkpoint.every_n == 2


def test_hash_ignores_labels_but_not_settings() -> None:
    base = load_spec_text(TORCH)
    relabelled = load_spec_text(TORCH.replace("name: mnist-cnn", "name: renamed")
                                .replace("tags: [baseline]", "tags: [other]"))
    assert spec_hash(base) == spec_hash(relabelled)
    assert spec_hash(base) != spec_hash(load_spec_text(TORCH.replace("lr: 0.001", "lr: 0.01")))
    assert spec_hash(base) != spec_hash(load_spec_text(TORCH.replace("epochs: 3", "epochs: 4")))
    assert len(spec_hash(base)) == 64


def test_specs_are_immutable() -> None:
    spec = load_spec_text(TORCH)
    with pytest.raises(ValidationError, match="frozen"):
        spec.seed = 7  # type: ignore[misc]
    assert spec.model_copy(update={"seed": 7}).seed == 7


def test_minimal_spec_gets_sensible_defaults() -> None:
    spec = ExperimentSpec.model_validate({"name": "x", "task": "image_classification",
                                          "data": {"dataset": "mnist"}, "model": {"model": "lenet5"},
                                          "training": {"kind": "torch"}})
    assert (spec.runtime.device, spec.runtime.precision, spec.seed) == ("auto", "fp32", 42)
    assert spec.training.optimizer.name == "adam" and spec.data.split.val_fraction == 0.1
