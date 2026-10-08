"""Experiments from the app: a real worker process trains, the lab database records it,
cancel and resume work."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.spec import (  # noqa: E402
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    RuntimeSpec,
    TorchTrainingSpec,
)
from tests.fixtures.fake_vision import write_fake_mnist  # noqa: E402


def spec(epochs: int = 2, steps: int = 3, name: str = "Fake MNIST · TinyVGG") -> ExperimentSpec:
    return ExperimentSpec(name=name, task=Task.IMAGE_CLASSIFICATION, data=DataSpec(dataset="mnist"),
                          model=ModelSpec(model="tiny_vgg"),
                          training=TorchTrainingSpec(epochs=epochs, batch_size=16, max_steps_per_epoch=steps),
                          runtime=RuntimeSpec(device="cpu"))


def wait_for(qtbot, ctx, run_id, statuses, timeout=180_000):
    qtbot.waitUntil(lambda: ctx.experiments.view(run_id).status in statuses, timeout=timeout)
    return ctx.experiments.view(run_id)


def test_missing_data_is_reported_before_queueing(ctx) -> None:
    problems = ctx.experiments.problems(spec())
    assert problems and "MNIST is not downloaded yet" in problems[0]


def test_a_run_is_recorded_from_start_to_finish(ctx, qtbot) -> None:
    write_fake_mnist(ctx.paths.datasets)
    service = ctx.experiments
    assert service.problems(spec()) == []
    epochs = []
    service.run_epoch.connect(lambda run_id, epoch, metrics: epochs.append(epoch))
    run_id = service.launch(spec())
    view = service.view(run_id)
    assert view.status in ("queued", "running") and view.model == "tiny_vgg" and view.epochs == 2
    assert view.run_dir.parent.name.startswith("fake-mnist-tinyvgg-")
    assert (view.run_dir / "experiment.yaml").is_file()
    done = wait_for(qtbot, ctx, run_id, ("completed", "failed"))
    assert done.status == "completed", done.error
    assert done.device == "cpu" and done.duration_s and epochs == [1, 2]
    history = service.history(run_id, ("val_acc", "train_loss"))
    assert [epoch for epoch, _ in history["val_acc"]] == [1, 2]
    assert "test_acc" in ctx.store.latest_metrics(run_id)
    kinds = {row["kind"] for row in ctx.store.artifacts(run_id)}
    assert {"checkpoint", "confusion_matrix", "explainer_weights"} <= kinds
    assert any("Test accuracy" in line for line in service.log_lines(run_id))
    assert service.live(run_id) is None and not done.resumable


def test_cancel_then_resume(ctx, qtbot) -> None:
    write_fake_mnist(ctx.paths.datasets)
    service = ctx.experiments
    run_id = service.launch(spec(epochs=40, steps=2, name="long"))
    with qtbot.waitSignal(service.run_epoch, timeout=180_000):
        pass
    assert service.live(run_id).epoch >= 1 and service.live(run_id).epochs == 40
    service.cancel(run_id)
    cancelled = wait_for(qtbot, ctx, run_id, ("cancelled", "failed"))
    assert cancelled.status == "cancelled" and cancelled.resumable
    done_before = len(service.history(run_id, ("train_loss",))["train_loss"])
    # Continue with fewer epochs left: rewrite the spec to stop soon after.
    short = spec(epochs=done_before + 1, steps=2, name="long")
    from core.experiment_engine.spec import dump_spec
    dump_spec(short, cancelled.run_dir / "experiment.yaml")
    service.resume(run_id)
    assert service.view(run_id).status in ("queued", "running")
    finished = wait_for(qtbot, ctx, run_id, ("completed", "failed"))
    assert finished.status == "completed", finished.error
    epochs = [epoch for epoch, _ in service.history(run_id, ("train_loss",))["train_loss"]]
    assert epochs == list(range(1, done_before + 2))
    with pytest.raises(ValueError):
        service.resume(run_id)


def test_the_first_run_is_recorded_as_started(ctx, qtbot) -> None:
    write_fake_mnist(ctx.paths.datasets)
    run_id = ctx.experiments.launch(spec(epochs=1, steps=1))
    assert ctx.experiments.view(run_id).started_at is not None  # it started inside launch()
    wait_for(qtbot, ctx, run_id, ("completed", "failed"))
