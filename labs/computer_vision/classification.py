"""The PyTorch image-classification runner: MNIST, Fashion-MNIST and CIFAR-10 with
SimpleCNN, LeNet-5, ResNet-18 or TinyVGG.

Runs inside the worker process. Each epoch trains, validates, saves ``last.pt`` (the
resume point) and, when the monitored metric improves, ``best.pt``. At the end the best
weights are tested on the held-out test split. Files go to the run folder:
``checkpoints/``, ``metrics.jsonl``, ``run_info.json``, ``test_confusion.json`` and,
for TinyVGG, the CNN Explainer's ``explainer.npz``.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.catalog import load_cards
from core.common.cancel import Cancelled
from core.common.optional import is_installed
from core.common.paths import AppPaths
from core.common.vocab import Task
from core.dataset_registry.adapter import DatasetAdapter
from core.experiment_engine.runner import ExperimentRunner, RunCallbacks, RunContext, RunResult
from core.experiment_engine.spec import ExperimentSpec

from .datasets import VisionAdapter
from .export import export_for_explainer
from .training.data import build_data, input_size_for, transform_problems
from .training.loop import (
    Mixed,
    Monitor,
    evaluate,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train_epoch,
)

__all__ = ["MONITORABLE", "TorchClassificationRunner"]

#: Metrics a checkpoint or early stopping can watch.
MONITORABLE = ("val_loss", "val_acc", "val_f1", "train_loss", "train_acc")


def _seed_everything(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


class TorchClassificationRunner(ExperimentRunner):
    id = "torch_classification"
    title = "PyTorch image classification"
    tasks = frozenset({Task.IMAGE_CLASSIFICATION})

    # Checks --------------------------------------------------------------------
    def _parts(self, spec: ExperimentSpec) -> tuple[AppPaths, VisionAdapter, Any, Any]:
        """Workspace paths, dataset adapter, model card and builder; raises with a message."""
        paths = AppPaths.resolve()
        datasets, models = load_cards(paths)
        if spec.data.dataset not in datasets:
            raise LookupError(f"no dataset card {spec.data.dataset!r}")
        adapter: DatasetAdapter = datasets.adapter(spec.data.dataset)
        if not isinstance(adapter, VisionAdapter):
            raise TypeError(f"{adapter.card.name} is not an image-classification dataset")
        if spec.model.model not in models:
            raise LookupError(f"no model card {spec.model.model!r}")
        card = models.get(spec.model.model)
        if Task.IMAGE_CLASSIFICATION not in card.tasks:
            raise ValueError(f"{card.name} is not an image-classification model")
        if spec.model.pretrained:
            card.weights_by_id(spec.model.pretrained)
        return paths, adapter, card, models.builder(spec.model.model)

    def validate(self, spec: ExperimentSpec) -> list[str]:
        problems = super().validate(spec)
        if problems:
            return problems
        if not is_installed("torch") or not is_installed("torchvision"):
            problems.append("PyTorch and torchvision are needed in the experiment Python environment")
        try:
            paths, adapter, _card, _builder = self._parts(spec)
        except (LookupError, TypeError, ValueError, NotImplementedError) as exc:
            return [*problems, str(exc).strip("'\"")]
        if adapter.local_status(paths.datasets) != "present":
            problems.append(f"{adapter.card.name} is not downloaded yet: download it from the "
                            "Experiment Builder or the Computer Vision lab")
        problems += transform_problems(spec)
        training = spec.training
        watched = [training.checkpoint.monitor] + ([training.early_stopping.monitor]
                                                   if training.early_stopping else [])
        for key in watched:
            if key not in MONITORABLE:
                problems.append(f"cannot monitor {key!r}; choose one of {', '.join(MONITORABLE)}")
            elif key.startswith("val_") and spec.data.split.val_fraction == 0:
                problems.append(f"{key} needs a validation split (data.split.val_fraction > 0)")
        return problems

    # Run -----------------------------------------------------------------------
    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        import torch
        from torch.utils.data import DataLoader

        started = time.monotonic()
        paths, adapter, card, builder = self._parts(spec)
        training = spec.training
        if spec.runtime.precision == "fp16" and not ctx.device.startswith("cuda"):
            raise ValueError("fp16 needs a CUDA GPU; use fp32 (or bf16) on this device")
        os.environ.setdefault("TORCH_HOME", str(paths.models / "torch-hub"))  # never ~/.cache
        _seed_everything(spec.seed, spec.runtime.deterministic)

        size = input_size_for(spec, adapter, builder)
        callbacks.on_log(f"Loading {adapter.card.name} at {size[0]}×{size[1]}")
        data = build_data(spec, adapter, paths.datasets, size)
        model = builder(num_classes=len(data.classes), in_channels=data.in_channels, input_size=size,
                        pretrained=spec.model.pretrained, **spec.model.params).to(ctx.device)
        parameters = sum(p.numel() for p in model.parameters())
        cuda = ctx.device.startswith("cuda")
        generator = torch.Generator().manual_seed(spec.seed)
        loader = dict(batch_size=training.batch_size, num_workers=spec.runtime.num_workers, pin_memory=cuda)
        train_loader = DataLoader(data.train, shuffle=True, generator=generator, **loader)
        val_loader = DataLoader(data.val, **loader) if data.val is not None else None
        test_loader = DataLoader(data.test, **loader)
        optimizer = make_optimizer(training, model.parameters())
        steps = min(len(train_loader), training.max_steps_per_epoch or len(train_loader))
        schedule = make_scheduler(training, optimizer, steps)
        mixed = Mixed(spec.runtime.precision, ctx.device)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)
        monitor = Monitor(training.checkpoint.monitor, training.checkpoint.mode)
        stopping = training.early_stopping
        stopper = Monitor(stopping.monitor, stopping.mode, stopping.min_delta) if stopping else None

        checkpoints = ctx.run_dir / "checkpoints"
        first_epoch = 1
        if ctx.resume_from is not None:
            done = load_checkpoint(ctx.resume_from, model=model, optimizer=optimizer, schedule=schedule,
                                   monitor=monitor)
            first_epoch = done + 1
            callbacks.on_log(f"Resuming after epoch {done} from {ctx.resume_from.name}")
        info = {"parameters": parameters, "input_size": list(size), "in_channels": data.in_channels,
                "classes": list(data.classes), "device": ctx.device, "torch": torch.__version__,
                "train_samples": len(data.train), "val_samples": len(data.val) if data.val else 0,
                "test_samples": len(data.test), "model": card.name, "dataset": adapter.card.name}
        (ctx.run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        ctx.tracker.log_params(info)
        callbacks.on_log(f"{card.name}: {parameters:,} parameters · {info['train_samples']:,} training, "
                         f"{info['val_samples']:,} validation, {info['test_samples']:,} test images "
                         f"on {ctx.device}")

        status, epoch = "completed", first_epoch - 1
        try:
            for epoch in range(first_epoch, training.epochs + 1):
                epoch_started = time.monotonic()
                callbacks.on_epoch_start(epoch, training.epochs)
                if cuda:
                    torch.cuda.reset_peak_memory_stats(ctx.device)
                metrics = train_epoch(model, train_loader, optimizer, loss_fn, ctx.device, mixed,
                                      schedule=schedule, cancel=ctx.cancel, on_step=callbacks.on_step,
                                      max_steps=training.max_steps_per_epoch,
                                      grad_clip=training.grad_clip_norm)
                if val_loader is not None:
                    val, _ = evaluate(model, val_loader, loss_fn, ctx.device, len(data.classes), mixed,
                                      cancel=ctx.cancel)
                    metrics.update({f"val_{key}": value for key, value in val.items()})
                metrics["lr"] = optimizer.param_groups[0]["lr"]
                metrics["epoch_time_s"] = time.monotonic() - epoch_started
                if cuda:
                    metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(ctx.device) / 1e9
                if schedule.every == "epoch":
                    schedule.scheduler.step()
                elif schedule.every == "plateau":
                    schedule.scheduler.step(metrics.get("val_loss", metrics["train_loss"]))
                improved = monitor.update(metrics, epoch)
                self._save(checkpoints, model, optimizer, schedule, epoch, monitor, spec, improved, callbacks)
                callbacks.on_epoch_end(epoch, metrics)
                ctx.tracker.log_metrics(metrics, step=epoch, epoch=epoch)
                _append_jsonl(ctx.run_dir / "metrics.jsonl", {"epoch": epoch, **metrics})
                if stopper is not None:
                    stopper.update(metrics, epoch)
                    if stopper.stale >= stopping.patience:
                        status = "early_stopped"
                        callbacks.on_log(f"Early stop: {stopper.key} has not improved for "
                                         f"{stopper.stale} epochs")
                        break
        except Cancelled:
            completed = max(epoch - 1, 0)
            callbacks.on_log(f"Cancelled; checkpoints/last.pt resumes after epoch {completed}")
            return RunResult(ctx.run_id, "cancelled", {"epochs_completed": float(max(epoch - 1, 0))},
                             duration_s=time.monotonic() - started)

        best = checkpoints / "best.pt"
        if best.is_file():
            model.load_state_dict(torch.load(best, map_location=ctx.device, weights_only=True)["model"])
        test, matrix = evaluate(model, test_loader, loss_fn, ctx.device, len(data.classes), mixed,
                                cancel=ctx.cancel)
        results = {f"test_{key}": value for key, value in test.items()}
        results.update({"epochs_completed": float(epoch), f"best_{monitor.key}": float(monitor.best or 0.0),
                        "best_epoch": float(monitor.best_epoch)})
        confusion = ctx.run_dir / "test_confusion.json"
        confusion.write_text(json.dumps({"classes": list(data.classes), "matrix": matrix.tolist()}),
                             encoding="utf-8")
        callbacks.on_artifact(confusion, "confusion_matrix")
        if spec.model.model == "tiny_vgg":
            exported = export_for_explainer(model.state_dict(), ctx.run_dir, dataset=spec.data.dataset,
                                            in_channels=data.in_channels, size=size[0],
                                            class_names=data.classes,
                                            hidden=int(spec.model.params.get("hidden", 10)),
                                            normalization=data.normalization, metrics=results)
            callbacks.on_artifact(exported, "explainer_weights")
        callbacks.on_log(f"Test accuracy {results['test_acc']:.4f} · macro F1 {results['test_f1']:.4f}")
        return RunResult(ctx.run_id, status, results, best if best.is_file() else None,
                         time.monotonic() - started)

    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        import torch
        from torch.utils.data import DataLoader

        paths, adapter, _card, builder = self._parts(spec)
        size = input_size_for(spec, adapter, builder)
        data = build_data(spec, adapter, paths.datasets, size)
        dataset = data.test if split == "test" else data.val
        if dataset is None:
            raise ValueError("this experiment has no validation split (val_fraction is 0)")
        model = builder(num_classes=len(data.classes), in_channels=data.in_channels, input_size=size,
                        pretrained=None, **spec.model.params).to(ctx.device)
        model.load_state_dict(torch.load(checkpoint, map_location=ctx.device, weights_only=True)["model"])
        callbacks.on_log(f"Evaluating {checkpoint.name} on {len(dataset):,} {split} images")
        loader = DataLoader(dataset, batch_size=spec.training.batch_size,
                            num_workers=spec.runtime.num_workers)
        loss_fn = torch.nn.CrossEntropyLoss()
        metrics, matrix = evaluate(model, loader, loss_fn, ctx.device, len(data.classes),
                                   Mixed(spec.runtime.precision, ctx.device), cancel=ctx.cancel)
        return {"metrics": metrics, "classes": list(data.classes), "matrix": matrix.tolist()}

    @staticmethod
    def _save(folder: Path, model: Any, optimizer: Any, schedule: Any, epoch: int, monitor: Monitor,
              spec: ExperimentSpec, improved: bool, callbacks: RunCallbacks) -> None:
        strategy = spec.training.checkpoint
        extra = {"spec_name": spec.name}
        last = save_checkpoint(folder / "last.pt", model=model, optimizer=optimizer, schedule=schedule,
                               epoch=epoch, monitor=monitor, extra=extra)
        if epoch == 1:
            callbacks.on_artifact(last, "checkpoint")
        if improved and strategy.strategy in ("best", "best_and_last"):
            callbacks.on_artifact(save_checkpoint(folder / "best.pt", model=model, optimizer=optimizer,
                                                  schedule=schedule, epoch=epoch, monitor=monitor,
                                                  extra=extra), "checkpoint")
        if strategy.strategy == "every_n" and epoch % (strategy.every_n or 1) == 0:
            callbacks.on_artifact(save_checkpoint(folder / f"epoch_{epoch}.pt", model=model,
                                                  optimizer=optimizer, schedule=schedule, epoch=epoch,
                                                  monitor=monitor, extra=extra), "checkpoint")
