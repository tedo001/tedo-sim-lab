"""One epoch of training, one pass of evaluation, and the pieces around them: optimiser,
learning-rate schedule, mixed precision, the monitored metric and checkpoints."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.common.cancel import CancelToken
from core.experiment_engine.spec import TorchTrainingSpec

from .metrics import confusion_matrix, summary

__all__ = ["Mixed", "Monitor", "evaluate", "load_checkpoint", "make_optimizer", "make_scheduler",
           "save_checkpoint", "train_epoch"]

StepFn = Callable[[int, int, Mapping[str, float]], None]
_REPORT_EVERY_S = 0.5


def make_optimizer(training: TorchTrainingSpec, parameters: Any) -> Any:
    import torch

    spec = training.optimizer
    if spec.name == "sgd":
        return torch.optim.SGD(parameters, lr=spec.lr, momentum=spec.momentum,
                               weight_decay=spec.weight_decay)
    factory = torch.optim.AdamW if spec.name == "adamw" else torch.optim.Adam
    return factory(parameters, lr=spec.lr, weight_decay=spec.weight_decay)


@dataclass
class Schedule:
    scheduler: Any
    #: When it steps: after every batch, after every epoch, or on the validation loss.
    every: str  # "step" | "epoch" | "plateau" | "never"


def make_scheduler(training: TorchTrainingSpec, optimizer: Any, steps_per_epoch: int) -> Schedule:
    from torch.optim import lr_scheduler

    spec, params = training.scheduler, training.scheduler.params
    if spec.name == "step":
        return Schedule(lr_scheduler.StepLR(optimizer, step_size=int(params.get("step_size", 5)),
                                            gamma=float(params.get("gamma", 0.1))), "epoch")
    if spec.name == "cosine":
        return Schedule(lr_scheduler.CosineAnnealingLR(optimizer, T_max=training.epochs), "epoch")
    if spec.name == "plateau":
        return Schedule(lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(params.get("factor", 0.1)),
                                                       patience=int(params.get("patience", 2))), "plateau")
    if spec.name == "onecycle":
        return Schedule(lr_scheduler.OneCycleLR(optimizer, max_lr=training.optimizer.lr,
                                                total_steps=training.epochs * max(steps_per_epoch, 1)),
                        "step")
    return Schedule(None, "never")


class Mixed:
    """Mixed precision: fp16 with gradient scaling on CUDA, bf16 where supported, else fp32."""

    def __init__(self, precision: str, device: str) -> None:
        import torch

        self.device_type = device.split(":")[0]
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        self.enabled = self.dtype is not None
        scaled = precision == "fp16" and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if scaled else None

    def autocast(self) -> Any:
        import torch

        return torch.autocast(self.device_type, dtype=self.dtype, enabled=self.enabled)


def train_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any, device: str, mixed: Mixed, *,
                schedule: Schedule, cancel: CancelToken, on_step: StepFn,
                max_steps: int | None = None, grad_clip: float | None = None) -> dict[str, float]:
    import torch

    model.train()
    total = min(len(loader), max_steps) if max_steps else len(loader)
    seen = correct = 0
    loss_sum = 0.0
    last_report = 0.0
    for step, (images, labels) in enumerate(loader, start=1):
        if step > total:
            break
        cancel.raise_if_cancelled()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with mixed.autocast():
            logits = model(images)
            loss = loss_fn(logits, labels)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError("the loss became NaN or infinite; try a lower learning rate")
        if mixed.scaler is not None:
            mixed.scaler.scale(loss).backward()
            mixed.scaler.unscale_(optimizer)
        else:
            loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if mixed.scaler is not None:
            mixed.scaler.step(optimizer)
            mixed.scaler.update()
        else:
            optimizer.step()
        if schedule.every == "step":
            schedule.scheduler.step()
        batch = labels.size(0)
        seen += batch
        loss_sum += value * batch
        correct += int((logits.argmax(dim=1) == labels).sum())
        now = time.monotonic()
        if now - last_report >= _REPORT_EVERY_S or step == total:
            last_report = now
            on_step(step, total, {"train_loss": loss_sum / seen, "train_acc": correct / seen,
                                  "lr": optimizer.param_groups[0]["lr"]})
    return {"train_loss": loss_sum / max(seen, 1), "train_acc": correct / max(seen, 1)}


def evaluate(model: Any, loader: Any, loss_fn: Any, device: str, num_classes: int, mixed: Mixed, *,
             cancel: CancelToken) -> tuple[dict[str, float], np.ndarray]:
    """Loss, accuracy and macro precision / recall / F1, plus the confusion matrix."""
    import torch

    model.eval()
    targets, predictions = [], []
    loss_sum, seen = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            cancel.raise_if_cancelled()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with mixed.autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)
            loss_sum += float(loss) * labels.size(0)
            seen += labels.size(0)
            targets.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    matrix = confusion_matrix(np.concatenate(targets), np.concatenate(predictions), num_classes)
    return {"loss": loss_sum / max(seen, 1), **summary(matrix)}, matrix


class Monitor:
    """The metric that picks the best checkpoint and drives early stopping."""

    def __init__(self, key: str, mode: str, min_delta: float = 0.0) -> None:
        self.key, self.mode, self.min_delta = key, mode, min_delta
        self.best: float | None = None
        self.best_epoch = 0
        self.stale = 0

    def update(self, metrics: Mapping[str, float], epoch: int) -> bool:
        """Record this epoch; ``True`` when it is the best so far."""
        value = metrics.get(self.key)
        if value is None:
            raise KeyError(f"monitored metric {self.key!r} was not computed (have: {', '.join(metrics)})")
        better = (self.best is None or (value < self.best - self.min_delta if self.mode == "min"
                                        else value > self.best + self.min_delta))
        if better:
            self.best, self.best_epoch, self.stale = float(value), epoch, 0
        else:
            self.stale += 1
        return better

    def state(self) -> dict[str, Any]:
        return {"best": self.best, "best_epoch": self.best_epoch, "stale": self.stale}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.best, self.best_epoch, self.stale = state["best"], state["best_epoch"], state["stale"]


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, schedule: Schedule, epoch: int,
                    monitor: Monitor, extra: Mapping[str, Any]) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": schedule.scheduler.state_dict() if schedule.scheduler else None,
                "epoch": epoch, "monitor": monitor.state(), **extra}, partial)
    partial.replace(path)
    return path


def load_checkpoint(path: Path, *, model: Any, optimizer: Any, schedule: Schedule,
                    monitor: Monitor) -> int:
    """Restore everything saved by :func:`save_checkpoint`; return the epoch it ended."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if schedule.scheduler is not None and state.get("scheduler"):
        schedule.scheduler.load_state_dict(state["scheduler"])
    monitor.restore(state["monitor"])
    return int(state["epoch"])
