"""Fine-tuning a pretrained transformer encoder (DistilBERT, BERT, MiniLM, ...) for text
classification, on the same training loop as the image networks: the epochs, optimiser, schedule,
mixed precision, checkpoints, early stopping, cancel and resume of a ``torch`` experiment.

The pretrained model is the one downloaded through the Model Zoo or the Hugging Face plugin into
``<workspace>/models/huggingface/<owner>__<name>/``; nothing is fetched during a run. A fresh
classification head with one output per class is put on top. Files in the run folder, beside the
usual ones: ``checkpoints/last.pt`` (the resume point) and ``checkpoints/best/`` (the best model and
its tokenizer in Hugging Face's format, which the Model Testing page and ``from_pretrained`` load)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from core.common.cancel import Cancelled
from core.common.optional import is_installed
from core.common.vocab import Task
from core.experiment_engine.runner import ExperimentRunner, RunCallbacks, RunContext, RunResult
from core.experiment_engine.spec import ExperimentSpec, TorchTrainingSpec

from ..computer_vision.training.data import split_indices
from ..computer_vision.training.loop import (
    Mixed,
    Monitor,
    evaluate,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train_epoch,
)
from ..inference.hub import is_downloaded, model_folder
from .text_data import load_splits, split_samples, text_adapter

__all__ = ["TransformerTextRunner", "hub_repo"]

BEST = "best"


def hub_repo(card: Any, weights_id: str | None) -> str:
    """The Hub repository of the card's chosen (or first) pretrained weights."""
    choices = [w for w in card.weights if getattr(w, "hub_repo", None)]
    if weights_id:
        choices = [w for w in choices if w.id == weights_id]
    if not choices:
        raise ValueError(f"{card.name} has no downloadable pretrained weights to fine-tune")
    return str(choices[0].hub_repo)


class _Tokens:
    """Token ids per text (truncated, not padded) and labels, as a torch-style dataset."""

    def __init__(self, ids: list[list[int]], labels: Any) -> None:
        self.ids, self.targets = ids, labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.ids[index], int(self.targets[index])


def _collate(pad: int) -> Any:
    def collate(batch: list[tuple[list[int], int]]) -> tuple[Any, Any]:
        import torch

        longest = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), longest), pad, dtype=torch.long)
        for row, (tokens, _label) in enumerate(batch):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return ids, torch.tensor([label for _, label in batch], dtype=torch.long)
    return collate


def _wrap(model: Any, pad: int) -> Any:
    """The Hugging Face model as a module that takes token ids and returns logits."""
    import torch

    class TokenClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hf = model

        def forward(self, ids: Any) -> Any:
            return self.hf(input_ids=ids, attention_mask=(ids != pad).long()).logits

    return TokenClassifier()


class TransformerTextRunner(ExperimentRunner):
    id = "text_transformer"
    title = "Text classification (fine-tuned transformer)"
    tasks = frozenset({Task.TEXT_CLASSIFICATION})

    def supports(self, spec: ExperimentSpec) -> bool:
        return spec.task in self.tasks and spec.training.kind == "torch"

    def _parts(self, spec: ExperimentSpec) -> tuple[Any, Any, Any, Path]:
        paths, adapter, card = text_adapter(spec)
        if card.framework.value != "huggingface":
            raise ValueError(f"{card.name} is not a transformer; TF-IDF models train with training.kind "
                             "'sklearn'")
        return paths, adapter, card, model_folder(paths.workspace, hub_repo(card, spec.model.pretrained))

    def validate(self, spec: ExperimentSpec) -> list[str]:
        problems = super().validate(spec)
        if problems:
            return problems
        from core.common.packs import pack_python

        if not is_installed("torch"):
            problems.append("torch is needed in the experiment Python environment")
        if not is_installed("transformers") and pack_python("language") is None:
            problems.append("transformers is needed: install the Language environment (Plugin Store → "
                            "Environments)")
        try:
            paths, adapter, card, folder = self._parts(spec)
        except (LookupError, TypeError, ValueError, NotImplementedError) as exc:
            return [*problems, str(exc).strip("'\"")]
        if adapter.local_status(paths.datasets) != "present":
            problems.append(f"{adapter.card.name} is not downloaded yet: download it from the Dataset Hub")
        if not is_downloaded(paths.workspace, hub_repo(card, spec.model.pretrained)):
            problems.append(f"{card.name} is not downloaded yet: download its weights in the Model Zoo")
        unknown = set(spec.model.params) - {"max_length"}
        if unknown:
            problems.append(f"model.params: unknown option(s) {', '.join(sorted(unknown))} (known: "
                            "max_length)")
        return problems

    def _data(self, spec: ExperimentSpec, adapter: Any, root: Path, tokenizer: Any) -> dict[str, Any]:
        from torch.utils.data import Subset

        data = load_splits(spec, adapter, root, 0.2)
        length = int(spec.model.params.get("max_length", 256))
        encode = lambda texts: tokenizer(texts, truncation=True, max_length=length)["input_ids"]  # noqa: E731
        train = _Tokens(encode(data.train), data.y_train)
        test = _Tokens(encode(data.test), data.y_test)
        train_idx, val_idx = split_indices(data.y_train, spec.data.split.val_fraction,
                                           stratify=spec.data.split.stratify, seed=spec.seed)
        return {"train": Subset(train, train_idx.tolist()),
                "val": Subset(train, val_idx.tolist()) if len(val_idx) else None,
                "test": test, "classes": data.classes}

    def _load(self, folder: Path, classes: list[str], device: str) -> tuple[Any, Any, Any]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(folder)
        hf = AutoModelForSequenceClassification.from_pretrained(
            folder, num_labels=len(classes), id2label=dict(enumerate(classes)),
            label2id={name: i for i, name in enumerate(classes)}, ignore_mismatched_sizes=True)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return tokenizer, hf, _wrap(hf, pad).to(device)

    def run(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext) -> RunResult:
        import torch
        from torch.utils.data import DataLoader

        started = time.monotonic()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        paths, adapter, card, folder = self._parts(spec)
        training = spec.training
        assert isinstance(training, TorchTrainingSpec)
        torch.manual_seed(spec.seed)
        classes = list(adapter.classes)
        tokenizer, hf, model = self._load(folder, classes, ctx.device)
        data = self._data(spec, adapter, paths.datasets, tokenizer)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = dict(batch_size=training.batch_size, collate_fn=_collate(pad))
        train_loader = DataLoader(data["train"], shuffle=True,
                                  generator=torch.Generator().manual_seed(spec.seed), **loader)
        val_loader = DataLoader(data["val"], **loader) if data["val"] is not None else None
        test_loader = DataLoader(data["test"], **loader)
        optimizer = make_optimizer(training, model.parameters())
        schedule = make_scheduler(training, optimizer, min(len(train_loader),
                                                           training.max_steps_per_epoch or len(train_loader)))
        mixed = Mixed(spec.runtime.precision, ctx.device)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)
        monitor = Monitor(training.checkpoint.monitor, training.checkpoint.mode)
        stopping = training.early_stopping
        stopper = Monitor(stopping.monitor, stopping.mode, stopping.min_delta) if stopping else None
        checkpoints = ctx.run_dir / "checkpoints"
        first = 1
        if ctx.resume_from is not None:
            first = load_checkpoint(ctx.resume_from, model=model, optimizer=optimizer, schedule=schedule,
                                    monitor=monitor) + 1
            callbacks.on_log(f"Resuming after epoch {first - 1}")
        parameters = sum(p.numel() for p in model.parameters())
        info = {"model": card.name, "dataset": adapter.card.name, "classes": classes,
                "parameters": parameters,
                "pretrained": folder.name, "device": ctx.device, "torch": torch.__version__,
                "train_samples": len(data["train"]), "val_samples": len(data["val"] or []),
                "test_samples": len(data["test"]), "max_length": int(spec.model.params.get("max_length",
                                                                                           256))}
        (ctx.run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        ctx.tracker.log_params(info)
        callbacks.on_log(f"{card.name}: {parameters:,} parameters · {info['train_samples']:,} training, "
                         f"{info['val_samples']:,} validation, {info['test_samples']:,} test texts on "
                         f"{ctx.device}")
        status, epoch = "completed", first - 1
        try:
            for epoch in range(first, training.epochs + 1):
                began = time.monotonic()
                callbacks.on_epoch_start(epoch, training.epochs)
                metrics = train_epoch(model, train_loader, optimizer, loss_fn, ctx.device, mixed,
                                      schedule=schedule, cancel=ctx.cancel, on_step=callbacks.on_step,
                                      max_steps=training.max_steps_per_epoch,
                                      grad_clip=training.grad_clip_norm)
                if val_loader is not None:
                    val, _ = evaluate(model, val_loader, loss_fn, ctx.device, len(classes), mixed,
                                      cancel=ctx.cancel)
                    metrics.update({f"val_{key}": value for key, value in val.items()})
                metrics.update(lr=optimizer.param_groups[0]["lr"], epoch_time_s=time.monotonic() - began)
                if schedule.every == "epoch":
                    schedule.scheduler.step()
                elif schedule.every == "plateau":
                    schedule.scheduler.step(metrics.get("val_loss", metrics["train_loss"]))
                if monitor.update(metrics, epoch):
                    hf.save_pretrained(checkpoints / BEST)
                    tokenizer.save_pretrained(checkpoints / BEST)
                    callbacks.on_artifact(checkpoints / BEST, "checkpoint")
                save_checkpoint(checkpoints / "last.pt", model=model, optimizer=optimizer, schedule=schedule,
                                epoch=epoch, monitor=monitor, extra={"spec_name": spec.name})
                callbacks.on_epoch_end(epoch, metrics)
                ctx.tracker.log_metrics(metrics, step=epoch, epoch=epoch)
                with (ctx.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
                if stopper is not None:
                    stopper.update(metrics, epoch)
                    if stopper.stale >= stopping.patience:
                        status = "early_stopped"
                        callbacks.on_log(f"Early stop: {stopper.key} has not improved for {stopper.stale} "
                                         "epochs")
                        break
        except Cancelled:
            callbacks.on_log(f"Cancelled; checkpoints/last.pt resumes after epoch {max(epoch - 1, 0)}")
            return RunResult(ctx.run_id, "cancelled", {"epochs_completed": float(max(epoch - 1, 0))},
                             duration_s=time.monotonic() - started)
        best = checkpoints / BEST
        if (best / "config.json").is_file():
            _tok, _hf, model = self._load(best, classes, ctx.device)
        test, matrix = evaluate(model, test_loader, loss_fn, ctx.device, len(classes), mixed,
                                cancel=ctx.cancel)
        results = {f"test_{key}": value for key, value in test.items()}
        results.update({"epochs_completed": float(epoch), f"best_{monitor.key}": float(monitor.best or 0.0),
                        "best_epoch": float(monitor.best_epoch)})
        confusion = ctx.run_dir / "test_confusion.json"
        confusion.write_text(json.dumps({"classes": classes, "matrix": matrix.tolist()}), encoding="utf-8")
        callbacks.on_artifact(confusion, "confusion_matrix")
        callbacks.on_log(f"Test accuracy {results['test_acc']:.4f} · macro F1 {results['test_f1']:.4f}")
        return RunResult(ctx.run_id, status, results, best if best.is_dir() else None,
                         time.monotonic() - started)

    def data_samples(self, spec: ExperimentSpec) -> dict[str, Any]:
        paths, adapter, _card, _folder = self._parts(spec)
        return split_samples(spec, load_splits(spec, adapter, paths.datasets, 0.2))

    def evaluate(self, spec: ExperimentSpec, callbacks: RunCallbacks, ctx: RunContext, checkpoint: Path,
                 split: str) -> dict[str, object]:
        import torch
        from torch.utils.data import DataLoader

        paths, adapter, _card, folder = self._parts(spec)
        classes = list(adapter.classes)
        tokenizer, _hf, model = self._load(folder, classes, ctx.device)
        if checkpoint.is_dir():
            tokenizer, _hf, model = self._load(checkpoint, classes, ctx.device)
        else:
            model.load_state_dict(torch.load(checkpoint, map_location=ctx.device, weights_only=True)["model"])
        data = self._data(spec, adapter, paths.datasets, tokenizer)
        dataset = data["test"] if split == "test" else data["val"]
        if dataset is None:
            raise ValueError("this experiment has no validation split (val_fraction is 0)")
        callbacks.on_log(f"Evaluating {checkpoint.name} on {len(dataset):,} {split} texts")
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = DataLoader(dataset, batch_size=spec.training.batch_size, collate_fn=_collate(pad))
        metrics, matrix = evaluate(model, loader, torch.nn.CrossEntropyLoss(), ctx.device, len(classes),
                                   Mixed(spec.runtime.precision, ctx.device), cancel=ctx.cancel)
        return {"metrics": metrics, "classes": classes, "matrix": matrix.tolist()}
