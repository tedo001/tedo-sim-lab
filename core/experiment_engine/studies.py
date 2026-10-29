"""Groups of runs that answer one question together.

- A **seed study** runs one experiment with several seeds, so a score is reported as a mean with
  its spread and interval instead of one lucky number. Its runs carry the tags ``study:<id>`` and
  ``seed:<n>``.
- A **sweep** runs one experiment over a hyperparameter space (every combination, or a random
  sample of them), for hyperparameter search and for seeing which parameter matters. Its runs carry
  ``sweep:<id>``; each run's values are read back from its own spec by the swept paths.

The definition of each group is kept in ``<workspace>/results/studies/<id>.json``. Every run's
**compute budget** (epochs, batches per epoch, samples per split, batch size, samples seen) is
recorded so two configurations are only compared on matched budgets; **baselines** are saved
configurations (``<workspace>/configs/baselines/<name>.yaml``) a study is measured against."""

from __future__ import annotations

import itertools
import json
import random
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .spec import ExperimentSpec, dump_spec, load_spec

__all__ = ["BASELINES", "STUDIES", "budget", "budget_mismatch", "list_baselines", "load_baseline",
           "save_baseline", "seed_study", "set_path", "get_path", "sweep", "write_definition"]

STUDIES = "studies"          # under results/
BASELINES = "baselines"      # under configs/
_NAME = re.compile(r"[^a-z0-9_-]+")


def get_path(data: Mapping[str, Any], path: str) -> Any:
    """``data["training"]["optimizer"]["lr"]`` for ``"training.optimizer.lr"``; None when absent."""
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _copy(spec: ExperimentSpec, changes: Mapping[str, Any], tags: Sequence[str],
          suffix: str) -> ExperimentSpec:
    data = spec.model_dump(mode="json")
    for path, value in changes.items():
        set_path(data, path, value)
    data["tags"] = [tag for tag in data.get("tags", []) if not tag.startswith(("study:", "sweep:", "seed:"))]
    data["tags"] += list(tags)
    data["name"] = f"{spec.name}-{suffix}"
    return ExperimentSpec.model_validate(data)


def seed_study(spec: ExperimentSpec, seeds: Sequence[int], study_id: str) -> list[ExperimentSpec]:
    """One spec per seed; everything else identical."""
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("give each seed once (at least one)")
    return [_copy(spec, {"seed": int(seed)}, (f"study:{study_id}", f"seed:{seed}"), f"seed{seed}")
            for seed in seeds]


def sweep(spec: ExperimentSpec, space: Mapping[str, Sequence[Any]], sweep_id: str, *,
          method: str = "grid", count: int | None = None, seed: int = 0) -> list[ExperimentSpec]:
    """One spec per point of ``space`` (path → values): every combination for ``grid``, ``count``
    distinct random combinations for ``random``. Paths are checked by building every spec."""
    if not space:
        raise ValueError("the sweep space is empty: give at least one path and its values")
    paths = list(space)
    for path in paths:
        if not isinstance(space[path], Sequence) or isinstance(space[path], str) or not space[path]:
            raise ValueError(f"{path}: give a list of values")
    points = list(itertools.product(*(space[path] for path in paths)))
    if method == "random":
        if not count or count < 1:
            raise ValueError("a random sweep needs a count of runs")
        points = random.Random(seed).sample(points, min(count, len(points)))
    elif method != "grid":
        raise ValueError(f"unknown sweep method {method!r} (grid or random)")
    out = []
    for index, point in enumerate(points, start=1):
        try:
            out.append(_copy(spec, dict(zip(paths, point, strict=True)), (f"sweep:{sweep_id}",), f"s{index}"))
        except ValueError as exc:
            raise ValueError(f"point {dict(zip(paths, point, strict=True))}: {exc}") from None
    return out


def write_definition(results: Path, group_id: str, kind: str, base: ExperimentSpec,
                     **details: Any) -> Path:
    folder = results / STUDIES
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{group_id}.json"
    path.write_text(json.dumps({"id": group_id, "kind": kind, "name": base.name, "task": base.task.value,
                                "dataset": base.data.dataset, "model": base.model.model,
                                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                                **details}, indent=2, default=str), encoding="utf-8")
    return path


def budget(spec: ExperimentSpec, run_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The compute a run was given: what was allowed (spec) and what it used (run_info)."""
    training = spec.training
    info = dict(run_info or {})
    out: dict[str, Any] = {"kind": training.kind, "subset": spec.data.subset}
    if training.kind in ("torch", "quantum"):
        out.update(epochs=training.epochs, batch_size=training.batch_size,
                   max_steps_per_epoch=getattr(training, "max_steps_per_epoch", None))
    else:
        search = getattr(training, "search", None)
        out.update(fits=1 if search is None else (search.n_iter or "grid"),
                   cv_folds=getattr(training, "cv_folds", None))
    train = info.get("train_samples") or info.get("train_rows")
    if train:
        out["train_samples"] = int(train)
        epochs = int(info.get("epochs_run") or out.get("epochs") or 1)
        steps = out.get("max_steps_per_epoch")
        seen = int(train) * epochs
        if steps and out.get("batch_size"):
            seen = min(seen, int(steps) * int(out["batch_size"]) * epochs)
        out["samples_seen"] = seen
    return out


def budget_mismatch(first: Mapping[str, Any], second: Mapping[str, Any],
                    tolerance: float = 0.05) -> list[str]:
    """How two budgets differ, in words; ``[]`` when they match (samples seen within ``tolerance``)."""
    notes = []
    for key in ("epochs", "max_steps_per_epoch", "subset", "batch_size", "cv_folds", "fits"):
        if first.get(key) != second.get(key):
            notes.append(f"{key}: {first.get(key)} against {second.get(key)}")
    a, b = first.get("samples_seen"), second.get("samples_seen")
    if a and b and abs(a - b) > tolerance * max(a, b):
        notes.append(f"samples seen: {a:,} against {b:,}")
    return notes


def _slug(name: str) -> str:
    return _NAME.sub("-", name.strip().lower()).strip("-") or "baseline"


def save_baseline(configs: Path, name: str, spec: ExperimentSpec) -> Path:
    """Keep ``spec`` as a named baseline configuration."""
    return dump_spec(spec, configs / BASELINES / f"{_slug(name)}.yaml")


def list_baselines(configs: Path) -> list[tuple[str, Path]]:
    folder = configs / BASELINES
    return [(path.stem, path) for path in sorted(folder.glob("*.yaml"))] if folder.is_dir() else []


def load_baseline(configs: Path, name: str) -> ExperimentSpec:
    return load_spec(configs / BASELINES / f"{_slug(name)}.yaml")
