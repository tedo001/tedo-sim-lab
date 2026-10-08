"""The PyTorch classification runner, end to end on a tiny fake MNIST."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from core.common.cancel import CancelToken  # noqa: E402
from core.common.paths import AppPaths  # noqa: E402
from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.events import JsonLinesCallbacks, parse_line  # noqa: E402
from core.experiment_engine.spec import (  # noqa: E402
    DataSpec,
    EarlyStoppingSpec,
    ExperimentSpec,
    ModelSpec,
    RuntimeSpec,
    TorchTrainingSpec,
    TransformSpec,
    dump_spec,
)
from core.experiment_engine.worker import run_experiment  # noqa: E402
from labs.computer_vision.classification import TorchClassificationRunner  # noqa: E402
from labs.computer_vision.explainer import ExplainerNet, run, tiny_vgg  # noqa: E402
from labs.computer_vision.export import fold_normalization  # noqa: E402
from labs.computer_vision.models import (  # noqa: E402
    build_lenet5,
    build_resnet18,
    build_simple_cnn,
    build_tiny_vgg,
)
from tests.fixtures.fake_vision import write_fake_mnist  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    write_fake_mnist(paths.datasets)
    return paths


def make_spec(model: str = "simple_cnn", **training) -> ExperimentSpec:
    options = {"epochs": 2, "batch_size": 32, **training}
    return ExperimentSpec(name=f"fake-{model}", task=Task.IMAGE_CLASSIFICATION,
                          data=DataSpec(dataset="mnist", preprocessing=(TransformSpec(name="normalize"),)),
                          model=ModelSpec(model=model), training=TorchTrainingSpec(**options),
                          runtime=RuntimeSpec(device="cpu"))


class Recorder(JsonLinesCallbacks):
    def __init__(self) -> None:
        self.lines: list[str] = []
        super().__init__(self, "run")  # type: ignore[arg-type]

    def write(self, text: str) -> None:
        self.lines += [line for line in text.splitlines() if line]

    def flush(self) -> None:
        pass

    def events(self, kind: str) -> list:
        return [event for event in map(parse_line, self.lines) if event.type == kind]


def train(paths: AppPaths, spec: ExperimentSpec, name: str = "run", **kwargs):
    run_dir = paths.experiments / name
    dump_spec(spec, run_dir / "experiment.yaml")
    recorder = Recorder()
    result = run_experiment(run_dir, recorder, kwargs.pop("token", CancelToken()), **kwargs)
    return run_dir, recorder, result


@pytest.mark.parametrize("builder", [build_simple_cnn, build_lenet5, build_resnet18, build_tiny_vgg])
@pytest.mark.parametrize("shape", [(1, 28, 28), (3, 32, 32)])
def test_builders_adapt_to_the_data(builder, shape) -> None:
    model = builder(num_classes=10, in_channels=shape[0], input_size=shape[1:], pretrained=None)
    assert model(torch.zeros(2, *shape)).shape == (2, 10)


def test_builders_refuse_what_they_cannot_do() -> None:
    with pytest.raises(ValueError, match="no pretrained"):
        build_simple_cnn(num_classes=10, in_channels=1, input_size=(28, 28), pretrained="imagenet1k_v1")
    with pytest.raises(ValueError, match="at least 16"):
        build_lenet5(num_classes=10, in_channels=1, input_size=(8, 8))
    assert build_lenet5.input_size == (32, 32)


def test_validate_explains_problems(workspace) -> None:
    runner = TorchClassificationRunner()
    assert runner.validate(make_spec()) == []
    missing = make_spec().model_copy(update={"data": DataSpec(dataset="cifar10")})
    assert "CIFAR-10 is not downloaded" in runner.validate(missing)[0]
    odd = make_spec().model_copy(update={"data": DataSpec(
        dataset="mnist", augmentation=(TransformSpec(name="mixup"),))})
    assert any("unknown augmentation 'mixup'" in p for p in runner.validate(odd))
    planned = make_spec(model="rt_detr")
    assert runner.validate(planned)


def test_training_end_to_end(workspace) -> None:
    run_dir, recorder, result = train(workspace, make_spec("tiny_vgg", epochs=3, max_steps_per_epoch=4))
    assert result.status == "completed", result.error
    epochs = recorder.events("epoch_end")
    assert [e.payload["epoch"] for e in epochs] == [1, 2, 3]
    assert {"train_loss", "val_loss", "val_acc", "val_f1", "lr"} <= set(epochs[0].payload["metrics"])
    assert recorder.events("step") and 0 <= result.metrics["test_acc"] <= 1
    assert (run_dir / "checkpoints" / "last.pt").is_file() and result.best_checkpoint.is_file()
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 3
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["classes"][0] == "0" and info["test_samples"] == 60
    confusion = json.loads((run_dir / "test_confusion.json").read_text())
    assert np.asarray(confusion["matrix"]).sum() == 60
    # The explainer gets the same network, with the normalisation folded into conv_1_1.
    meta = json.loads((run_dir / "explainer.json").read_text())
    net = ExplainerNet.load_npz(tiny_vgg(in_channels=1, size=28, class_names=tuple(meta["class_names"])),
                                run_dir / "explainer.npz")
    model = build_tiny_vgg(num_classes=10, in_channels=1, input_size=(28, 28))
    model.load_state_dict(torch.load(result.best_checkpoint, weights_only=True)["model"])
    image = np.random.default_rng(1).uniform(size=(1, 28, 28))
    expected = model.eval()(torch.tensor((image - 0.1307) / 0.3081, dtype=torch.float32)[None])[0]
    np.testing.assert_allclose(run(net, image).logits, expected.detach().numpy(), atol=1e-4)


def test_early_stopping_cancel_and_resume(workspace) -> None:
    stopping = EarlyStoppingSpec(monitor="train_loss", patience=1, min_delta=100.0)  # never improves
    _, recorder, result = train(workspace, make_spec(epochs=5, max_steps_per_epoch=2,
                                                     early_stopping=stopping), "stop")
    assert result.status == "early_stopped" and len(recorder.events("epoch_end")) == 2

    token = CancelToken()
    token.cancel()
    run_dir, _, cancelled = train(workspace, make_spec(epochs=2, max_steps_per_epoch=2), "resume",
                                  token=token)
    assert cancelled.status == "cancelled"
    _, first, done = train(workspace, make_spec(epochs=1, max_steps_per_epoch=2), "resume")
    assert done.status == "completed"
    spec = make_spec(epochs=3, max_steps_per_epoch=2)
    dump_spec(spec, run_dir / "experiment.yaml")
    recorder = Recorder()
    resumed = run_experiment(run_dir, recorder, CancelToken(),
                             resume_from=run_dir / "checkpoints" / "last.pt")
    assert resumed.status == "completed"
    assert [e.payload["epoch"] for e in recorder.events("epoch_end")] == [2, 3]
    assert any("Resuming after epoch 1" in e.payload["line"] for e in recorder.events("log"))


def test_fp16_needs_a_gpu(workspace) -> None:
    spec = make_spec().model_copy(update={"runtime": RuntimeSpec(device="cpu", precision="fp16")})
    _, _, result = train(workspace, spec)
    assert result.status == "failed" and "fp16 needs a CUDA GPU" in result.error


def test_folding_normalisation_is_exact() -> None:
    rng = np.random.default_rng(0)
    architecture = tiny_vgg(in_channels=3, size=20, class_names=tuple("abc"))
    net = ExplainerNet.untrained(architecture, seed=4)
    mean, std = (0.4, 0.5, 0.6), (0.2, 0.25, 0.3)
    folded = ExplainerNet(architecture, fold_normalization(net.params, "conv_1_1", mean, std))
    image = rng.uniform(size=(3, 20, 20))
    normalised = (image - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    np.testing.assert_allclose(run(folded, image).logits, run(net, normalised).logits, atol=1e-10)


def test_worker_evaluates_a_checkpoint(workspace) -> None:
    import json as json_module

    from core.experiment_engine.worker import run_evaluation
    run_dir, _, result = train(workspace, make_spec(epochs=1, max_steps_per_epoch=2), "evaluated")
    assert result.status == "completed"
    recorder = Recorder()
    best = run_dir / "checkpoints" / "best.pt"
    evaluated = run_evaluation(run_dir, recorder, CancelToken(), checkpoint=best, split="val")
    assert evaluated.status == "completed" and "val_acc" in evaluated.metrics
    report = json_module.loads((run_dir / "evaluations" / "val-best.json").read_text())
    assert report["split"] == "val" and len(report["classes"]) == 10 and report["metrics"]["acc"] >= 0
    missing = run_evaluation(run_dir, Recorder(), CancelToken(), checkpoint=run_dir / "nope.pt", split="test")
    assert missing.status == "failed" and "no checkpoint" in missing.error
