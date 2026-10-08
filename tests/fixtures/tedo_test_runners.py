"""Runners for testing the worker and the job queue (never registered in the app).

Imported by the worker subprocess as ``tedo_test_runners:<Class>`` with this
folder on PYTHONPATH.
"""

from __future__ import annotations

import time

from core.common.vocab import Task
from core.experiment_engine.runner import ExperimentRunner, RunResult


class CountingRunner(ExperimentRunner):
    """Emits epochs and steps quickly; prints to stdout to prove the stream stays clean."""

    id = "test_counting"
    title = "Counting (test)"
    tasks = frozenset({Task.IMAGE_CLASSIFICATION})

    def run(self, spec, callbacks, ctx):
        started = time.monotonic()
        epochs = spec.training.epochs
        loss = 1.0
        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_start(epoch, epochs)
            for step in range(1, 4):
                ctx.cancel.raise_if_cancelled()
                print("noise a library printed to stdout")
                callbacks.on_step(step, 3, {"loss": loss})
                loss *= 0.9
            metrics = {"loss": loss, "val_accuracy": 0.5 + epoch / 100}
            callbacks.on_epoch_end(epoch, metrics)
            ctx.tracker.log_metrics(metrics, step=epoch, epoch=epoch)
        return RunResult(ctx.run_id, "completed", {"loss": loss, "epochs_completed": float(epochs)},
                         duration_s=time.monotonic() - started)


class WaitingRunner(ExperimentRunner):
    """Runs until cancelled (or 30 s), checking the token every 20 ms."""

    id = "test_waiting"
    title = "Waiting (test)"
    tasks = frozenset({Task.IMAGE_CLASSIFICATION})

    def run(self, spec, callbacks, ctx):
        callbacks.on_epoch_start(1, 1)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if ctx.cancel.wait(0.02):
                ctx.cancel.raise_if_cancelled()
        return RunResult(ctx.run_id, "completed")


class FailingRunner(ExperimentRunner):
    id = "test_failing"
    title = "Failing (test)"
    tasks = frozenset({Task.IMAGE_CLASSIFICATION})

    def run(self, spec, callbacks, ctx):
        callbacks.on_log("about to fail")
        raise RuntimeError("boom while using hf_abcdefghijklmnopqrstuvwxyz0123")
