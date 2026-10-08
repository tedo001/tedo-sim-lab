"""Time a model's inference at several batch sizes.

The model comes from a finished run (its best checkpoint, PyTorch or scikit-learn) or, for an
image model card, is built untrained at a given input size; latency does not depend on the
weights. Each batch size gets warm-up passes, then timed passes: mean, median and 95th
percentile latency, throughput, and on CUDA the peak memory. A batch that runs out of GPU
memory ends the sweep with a note instead of failing it.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core.catalog import load_cards
from core.common.paths import AppPaths
from core.experiment_engine.runner import RunCallbacks, RunContext
from core.experiment_engine.spec import load_spec

__all__ = ["DEFAULTS", "run_benchmark"]

DEFAULTS = {"batch_sizes": [1, 8, 32], "warmup": 5, "iterations": 30, "precision": "fp32"}


def _stats(seconds: list[float], batch: int) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000.0
    mean = float(ms.mean())
    return {"batch": batch, "mean_ms": mean, "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "throughput": batch / (mean / 1000.0)}


def _time(fn: Callable[[], Any], warmup: int, iterations: int, sync: Callable[[], None],
          ctx: RunContext) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    seconds = []
    for _ in range(iterations):
        ctx.cancel.raise_if_cancelled()
        started = time.perf_counter()
        fn()
        sync()
        seconds.append(time.perf_counter() - started)
    return seconds


def _run_dir(config: dict[str, Any], paths: AppPaths) -> Path:
    """The run folder named in ``config`` (relative to the workspace, or absolute)."""
    folder = Path(config["run_dir"])
    return folder if folder.is_absolute() else paths.workspace / folder


def _torch_model(config: dict[str, Any], paths: AppPaths) -> tuple[Any, tuple[int, int, int], str]:
    import torch

    _datasets, models = load_cards(paths)
    if "run_dir" in config:
        run_dir = _run_dir(config, paths)
        spec = load_spec(run_dir / "experiment.yaml")
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        shape = (int(info["in_channels"]), *map(int, info["input_size"]))
        model = models.builder(spec.model.model)(num_classes=len(info["classes"]), in_channels=shape[0],
                                                 input_size=shape[1:], pretrained=None, **spec.model.params)
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            checkpoint = run_dir / "checkpoints" / "last.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        return model, shape, f"{spec.name} ({checkpoint.name})"
    card_id = config["model"]
    shape = tuple(int(v) for v in config.get("input_shape", (3, 32, 32)))
    model = models.builder(card_id)(num_classes=int(config.get("num_classes", 10)), in_channels=shape[0],
                                    input_size=shape[1:], pretrained=None)
    return model, shape, f"{models.get(card_id).name} (untrained)"


def _bench_torch(config: dict[str, Any], callbacks: RunCallbacks, ctx: RunContext, paths: AppPaths) -> dict:
    import torch

    model, shape, label = _torch_model(config, paths)
    device = ctx.device
    cuda = device.startswith("cuda")
    if config["precision"] == "fp16" and not cuda:
        raise ValueError("fp16 needs a CUDA GPU; use fp32 on this device")
    dtype = torch.float16 if config["precision"] == "fp16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    parameters = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    rows, note = [], ""
    sizes = config["batch_sizes"]
    callbacks.on_log(f"{label}: {parameters:,} parameters, input {shape[0]}×{shape[1]}×{shape[2]}, "
                     f"on {device}")
    for step, batch in enumerate(sizes, start=1):
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            data = torch.randn(batch, *shape, device=device, dtype=dtype)
            with torch.inference_mode():
                seconds = _time(lambda data=data: model(data), config["warmup"], config["iterations"], sync,
                                ctx)
            row = _stats(seconds, batch)
            if cuda:
                row["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
        except torch.cuda.OutOfMemoryError:
            note = f"batch {batch} ran out of GPU memory; larger batches were skipped"
            callbacks.on_log(note)
            break
        rows.append(row)
        callbacks.on_step(step, len(sizes), {"throughput": row["throughput"]})
        callbacks.on_log(f"batch {batch}: {row['mean_ms']:.2f} ms · {row['throughput']:.0f} samples/s")
    return {"model": label, "framework": "pytorch", "parameters": parameters, "size_mb": size_mb,
            "input": list(shape), "torch": torch.__version__, "rows": rows, "note": note}


def _bench_sklearn(config: dict[str, Any], callbacks: RunCallbacks, ctx: RunContext, paths: AppPaths) -> dict:
    import joblib
    import sklearn

    from labs.classical_ml.runner import MODEL_FILE
    from labs.classical_ml.tables import load_table

    run_dir = _run_dir(config, paths)
    spec = load_spec(run_dir / "experiment.yaml")
    datasets, _models = load_cards(paths)
    table = load_table(spec, datasets.adapter(spec.data.dataset), paths.datasets)
    pipeline = joblib.load(run_dir / MODEL_FILE)  # the lab's own file from this run
    predict = pipeline.predict if spec.task.value != "dimensionality_reduction" else pipeline.transform
    rng = np.random.default_rng(spec.seed)
    rows, sizes = [], config["batch_sizes"]
    callbacks.on_log(f"{spec.name}: {spec.model.model} on {len(table.features):,} rows, CPU")
    for step, batch in enumerate(sizes, start=1):
        sample = table.features.iloc[rng.integers(0, len(table.features), batch)]
        seconds = _time(lambda s=sample: predict(s), config["warmup"], config["iterations"],
                        lambda: None, ctx)
        row = _stats(seconds, batch)
        rows.append(row)
        callbacks.on_step(step, len(sizes), {"throughput": row["throughput"]})
        callbacks.on_log(f"batch {batch}: {row['mean_ms']:.3f} ms · {row['throughput']:.0f} rows/s")
    return {"model": f"{spec.name} ({spec.model.model})", "framework": "sklearn", "parameters": None,
            "size_mb": (run_dir / MODEL_FILE).stat().st_size / 1e6, "input": [table.features.shape[1]],
            "sklearn": sklearn.__version__, "rows": rows, "note": ""}


def run_benchmark(config: dict[str, Any], callbacks: RunCallbacks, ctx: RunContext) -> dict[str, Any]:
    """Benchmark what ``config`` names; returns the results (the worker writes them to disk)."""
    config = {**DEFAULTS, **config}
    if not config["batch_sizes"] or any(int(b) < 1 for b in config["batch_sizes"]):
        raise ValueError("batch sizes must be whole numbers of 1 or more")
    config["batch_sizes"] = sorted({int(b) for b in config["batch_sizes"]})
    paths = AppPaths.resolve()
    tabular = "run_dir" in config and (_run_dir(config, paths) / "results.json").is_file()
    results = (_bench_sklearn if tabular else _bench_torch)(config, callbacks, ctx, paths)
    results.update({"device": "cpu" if tabular else ctx.device, "precision": config["precision"],
                    "warmup": config["warmup"], "iterations": config["iterations"],
                    "run_id": _run_dir(config, paths).name if "run_dir" in config else None,
                    "cpu": platform.processor() or platform.machine(), "python": platform.python_version()})
    return results
