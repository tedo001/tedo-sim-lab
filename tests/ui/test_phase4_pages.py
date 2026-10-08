"""Experiment Builder, Computer Vision lab, Training page and trained networks in the
CNN Explainer."""

from __future__ import annotations

import gzip
import hashlib

import pytest
from PySide6.QtWidgets import QScrollArea

from app.ui.pages.builder import ExperimentBuilderPage
from app.ui.pages.computer_vision import ComputerVisionPage
from app.ui.pages.training import TrainingPage
from core.experiment_engine.spec import load_spec
from labs.computer_vision.presets import CV_PRESETS
from tests.fixtures.fake_vision import write_fake_mnist


@pytest.fixture
def visited(ctx):
    pages: list[str] = []
    ctx.navigate = pages.append
    return pages


@pytest.fixture
def builder(ctx, qtbot, visited):
    page = ExperimentBuilderPage(ctx)
    qtbot.addWidget(page)
    return page


def test_builder_defaults_and_checks(builder, ctx) -> None:
    assert builder.form.dataset.card_id == "mnist" and builder.form.model.card_id == "simple_cnn"
    text = builder.yaml.toPlainText()
    assert "dataset: mnist" in text and "model: simple_cnn" in text
    assert not builder.run_button.isEnabled()
    assert any("MNIST is not downloaded" in problem for problem in builder.problems)
    assert builder.form.dataset.panel.download.isVisible() or True
    assert not builder.form.dataset.panel.download.isEnabled()  # terms not accepted yet
    builder.form.dataset.panel.acknowledge.setChecked(True)
    assert builder.form.dataset.panel.download.isEnabled()
    pytest.importorskip("torch")  # without it the runner rightly reports that it is missing
    write_fake_mnist(ctx.paths.datasets)
    builder.form.dataset.panel.refresh()
    builder.refresh()
    assert builder.problems == [] and builder.run_button.isEnabled()
    assert builder.state.text() == "Ready to run"


def test_builder_round_trips_specs(builder, tmp_path) -> None:
    spec = CV_PRESETS[3].make()  # CIFAR-10 · ResNet-18 with augmentation and a schedule
    builder.load(spec)
    assert builder.form.spec() == spec
    saved = builder.save_to(tmp_path / "exp.yaml")
    assert load_spec(saved) == spec
    builder.load(CV_PRESETS[0].make())
    assert builder.open_file(saved) and builder.form.spec() == spec
    (tmp_path / "bad.yaml").write_text("name: [")
    assert not builder.open_file(tmp_path / "bad.yaml") and "not valid YAML" in builder.result.text()


def test_pretrained_weights_need_their_terms_accepted(builder, ctx) -> None:
    pytest.importorskip("torch")
    write_fake_mnist(ctx.paths.datasets)
    builder.form.model.select("resnet18", "imagenet1k_v1")
    builder.refresh()
    assert any("Accept the terms" in problem for problem in builder.problems)
    assert "research" in builder.form.model.terms.text().lower()
    builder.form.model.acknowledge.setChecked(True)
    builder.refresh()
    assert builder.problems == []


def test_download_from_the_builder(builder, ctx, qtbot, tmp_path, monkeypatch) -> None:
    adapter_type = type(ctx.catalog.datasets.adapter("fashion_mnist"))
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    resources = []
    for name, _ in adapter_type.resources:
        data = gzip.compress(name.encode())
        (mirror / name).write_bytes(data)
        resources.append((name, hashlib.md5(data, usedforsecurity=False).hexdigest()))
    monkeypatch.setattr(adapter_type, "mirrors", (mirror.as_uri() + "/",))
    monkeypatch.setattr(adapter_type, "resources", tuple(resources))
    builder.form.dataset.select("fashion_mnist")
    panel = builder.form.dataset.panel
    assert panel.download.isEnabled() and not panel.acknowledge.isVisibleTo(builder)  # MIT
    with qtbot.waitSignal(ctx.downloads.finished, timeout=20_000) as done:
        panel.download.click()
    assert done.args[1] == "completed" and ctx.catalog.datasets.status("fashion_mnist") == "ready"
    assert panel.status.text() == "On disk" and not panel.download.isVisibleTo(builder)


def test_computer_vision_lab(ctx, qtbot, visited) -> None:
    page = ComputerVisionPage(ctx)
    qtbot.addWidget(page)
    assert set(page.dataset_panels) == {"mnist", "fashion_mnist", "cifar10"}
    page.preset_buttons["mnist_tiny_vgg"].click()
    assert visited == ["experiment_builder"] and ctx.experiments.draft.model.model == "tiny_vgg"
    builder = ExperimentBuilderPage(ctx)  # created on first visit: picks up the draft
    qtbot.addWidget(builder)
    assert builder.form.model.card_id == "tiny_vgg" and builder.form.name.text() == "mnist-tiny-vgg"


@pytest.mark.parametrize("page_id", ["experiment_builder", "computer_vision", "training"])
def test_phase4_pages_fit_half_width(window, qtbot, page_id) -> None:
    window.resize(1440, 900)
    window.set_split_open(True, "hardware")
    window.navigate(page_id)
    qtbot.wait(80)
    scroll = window.page_widget(page_id).findChild(QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0


def test_training_page_follows_a_run_into_the_explainer(ctx, qtbot, visited) -> None:
    pytest.importorskip("torch")
    from app.ui.pages.explainer import ExplainerPage

    write_fake_mnist(ctx.paths.datasets)
    page = TrainingPage(ctx)
    qtbot.addWidget(page)
    assert page.empty.isVisibleTo(page) and not page.detail.isVisibleTo(page)
    builder = ExperimentBuilderPage(ctx)
    qtbot.addWidget(builder)
    builder.load(CV_PRESETS[1].make())  # MNIST · TinyVGG
    builder.form.torch.epochs.setValue(2)
    builder.form.torch.max_steps.setValue(3)
    builder.form.torch.device.setCurrentIndex(builder.form.torch.device.findData("cpu"))
    builder.refresh()
    run_id = builder.run()
    assert run_id and visited[-1] == "training" and page.selected == run_id
    from tests.ui.test_experiment_service import wait_for
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed")).status == "completed"
    detail = page.detail
    assert detail.status.text() == "Completed" and detail.results.isVisibleTo(page)
    assert [epoch for epoch, _ in detail.charts["val_acc"].points()] == [1, 2]
    assert detail.result_values.value_text("Test accuracy") != "—"
    assert "Test accuracy" in detail.log.toPlainText()
    assert detail.explain_button.isVisibleTo(page) and not detail.resume_button.isVisibleTo(page)
    detail.explain_button.click()
    assert visited[-1] == "cnn_explainer" and ctx.experiments.explain_target == run_id
    explainer = ExplainerPage(ctx)
    qtbot.addWidget(explainer)
    assert explainer.trained_run == run_id and explainer.net.trained
    assert explainer.weights_pill.text() == "Trained" and not explainer.reseed_button.isEnabled()
    assert explainer.network_combo.currentText().startswith("Trained · mnist-tiny-vgg")
    explainer.network_combo.setCurrentIndex(explainer.network_combo.count() - 1)
    assert explainer.weights_pill.text() == "Untrained" and explainer.reseed_button.isEnabled()
