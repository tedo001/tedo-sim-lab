"""Dataset Hub (catalogue, Ontology Explorer, COCO inspector), Model Zoo and Model Registry."""

from __future__ import annotations

import pytest

from app.ui.pages.dataset_hub import DatasetHubPage
from app.ui.pages.model_registry import ModelRegistryPage
from app.ui.pages.model_zoo import ModelZooPage
from core.common.licensing import LicenseCategory
from tests.fixtures.fake_coco import write_fake_coco
from tests.ui.test_experiment_service import wait_for


def test_dataset_catalogue_filters_and_terms(ctx, qtbot) -> None:
    page = DatasetHubPage(ctx)
    qtbot.addWidget(page)
    catalogue = page.views["catalogue"]
    assert catalogue.table.rowCount() == len(ctx.catalog.datasets) == len(catalogue.ids)
    catalogue.filters.licence.setCurrentIndex(catalogue.filters.licence.findData(LicenseCategory.GATED))
    gated = {ctx.catalog.datasets.get(card_id).license.category for card_id in catalogue.ids}
    assert gated == {LicenseCategory.GATED} and "imagenet" in catalogue.ids
    catalogue.select("imagenet")
    assert not catalogue.download.isVisibleTo(catalogue) and catalogue.loader.isVisibleTo(catalogue)
    assert "account" in catalogue.values.value_text("Access")
    catalogue.filters.licence.setCurrentIndex(0)
    catalogue.filters.search.setText("mnist")
    assert set(catalogue.ids) >= {"mnist", "fashion_mnist"}
    catalogue.select("mnist")
    assert catalogue.download.isVisibleTo(catalogue) and catalogue.download.card_id == "mnist"
    assert catalogue.values.value_text("Splits").startswith("train 60,000")


def test_ontology_explorer_without_wordnet(ctx, qtbot) -> None:
    page = DatasetHubPage(ctx)
    qtbot.addWidget(page)
    view = page.show_view("ontology")
    assert page.switch.current == "ontology" and not page.views["catalogue"].isVisibleTo(page)
    assert view.download.isVisibleTo(view) and view.download.card_id == "wordnet"
    assert view.table.rowCount() == 200 and "Download WordNet" in view.state.text()
    view.search.setText("goldfish")
    assert view.keys == ["n01443537"] and view.detail.title.text() == "goldfish"
    assert view.detail.head.isVisibleTo(view)
    assert view.values.value_text("ImageNet class") == "1" and "orange-red" in view.gloss.text()
    assert "Download WordNet" in view.path.text()


def test_ontology_explorer_with_wordnet(ctx, qtbot) -> None:
    from tests.labs.test_ontology_coco import _installed_wordnet

    source = _installed_wordnet()
    if source is None:
        pytest.skip("WordNet is not installed for nltk on this machine")
    import shutil

    target = ctx.paths.datasets / "nltk_data" / "corpora"
    target.mkdir(parents=True)
    shutil.copy2(source, target / "wordnet.zip")
    page = DatasetHubPage(ctx)
    qtbot.addWidget(page)
    view = page.show_view("ontology")
    qtbot.waitUntil(lambda: view.wordnet is not None, timeout=60_000)
    assert not view.download.isVisibleTo(view) and "loaded" in view.state.text()
    view.search.setText("dog")
    assert "n02084071" in view.keys
    view.show_synset("n02084071")
    assert view.detail.title.text() == "dog" and "entity" in view.path.text()
    assert view.below.text().startswith("118:")
    view.path.linkActivated.emit("n01317541")  # domestic animal, from the path
    assert view.current == "n01317541"


def test_coco_inspector(ctx, qtbot, tmp_path) -> None:
    path = write_fake_coco(tmp_path / "coco", with_images=True)
    page = DatasetHubPage(ctx)
    qtbot.addWidget(page)
    view = page.show_view("coco")
    assert not view.stats.isVisibleTo(view) and not view.folder_button.isEnabled()
    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=30_000):
        view.open_file(path)
    assert view.tiles["images"].value.text() == "2" and view.tiles["annotations"].value.text() == "3"
    assert view.licences.rowCount() == 2 and view.category_bars.rows()[0][0] == "person"
    assert view.images.rowCount() == 2 and len(view.canvas.annotations) == 2
    assert view.canvas.image is None and "choose the images folder" in view.image_note.text()
    view.set_images_dir(tmp_path / "coco")
    assert view.canvas.image is not None and view.canvas.image.width() == 64
    view.category.setCurrentIndex(view.category.findData("car"))
    assert view.images.rowCount() == 1
    view.category.setCurrentIndex(0)
    view.images.selectRow(1)
    assert view.canvas.masks and "Attribution-NonCommercial" in view.image_note.text()


def test_model_zoo(ctx, qtbot) -> None:
    page = ModelZooPage(ctx)
    qtbot.addWidget(page)
    assert page.table.rowCount() == len(ctx.catalog.models)
    assert page.excluded.rowCount() >= 1 and "AGPL" in page.excluded.item(0, 1).text()
    page.filters.search.setText("resnet")
    assert set(page.ids) == {"resnet18", "detr_resnet50"}
    page.filters.search.setText("resnet-18")
    assert page.ids == ["resnet18"]
    assert page.weights.rowCount() == 1 and page.values.value_text("Code licence").startswith("BSD")
    page.filters.search.setText("")
    page.filters.first.setCurrentIndex(page.filters.first.findData(page.filters.first.itemData(2)))
    frameworks = {ctx.catalog.models.get(i).framework for i in page.ids}
    assert len(frameworks) <= 1


def _finished_classical_run(ctx, qtbot) -> str:
    pytest.importorskip("sklearn")
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    run_id = ctx.experiments.launch(CLASSICAL_PRESETS[0].make())  # Iris · logistic regression
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000).status == "completed"
    return run_id


def test_model_registry(ctx, qtbot) -> None:
    from app.ui.pages.training_run import RunDetail

    run_id = _finished_classical_run(ctx, qtbot)
    detail = RunDetail(ctx)
    qtbot.addWidget(detail)
    detail.show_run(run_id)
    assert detail.register_button.isVisibleTo(detail)
    page = ModelRegistryPage(ctx)
    qtbot.addWidget(page)
    assert page.empty.isVisibleTo(page) and not page.detail.isVisibleTo(page)
    detail.register_button.click()
    assert "Registered as iris-logistic-regression v1" in detail.registered.text()
    second = ctx.models.register(run_id)
    assert second.version == 2 and second.checkpoint.name == "model.joblib"
    assert "test_acc" in second.metrics and "cv_mean" in second.metrics
    assert page.table.rowCount() == 2 and page.detail.title.text() == "iris-logistic-regression v2"
    assert page.values.value_text("MLflow").startswith("not registered")
    page.stage.setCurrentIndex(page.stage.findData("production"))
    first = next(m for m in ctx.models.all() if m.version == 1)
    ctx.models.set_stage(first.id, "production")
    stages = {m.version: m.stage for m in ctx.models.all()}
    assert stages == {1: "production", 2: "archived"}
    page.notes.setPlainText("checked on the held-out rows")
    page._save_notes()
    assert ctx.models.get(page.selected).notes == "checked on the held-out rows"
    visited, shown = [], []
    ctx.navigate = visited.append
    ctx.experiments.show_requested.connect(shown.append)
    page._open_run()
    assert visited == ["training"] and shown == [run_id]
    assert page.delete(confirm=False) and page.table.rowCount() == 1


def test_registering_also_registers_in_mlflow(ctx, qtbot, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    monkeypatch.setenv("TEDO_LAB_MLFLOW", "1")
    run_id = _finished_classical_run(ctx, qtbot)
    assert ctx.experiments.view(run_id).mlflow_run_id
    with qtbot.waitSignal(ctx.models.mlflow_registered, timeout=120_000) as signal:
        model = ctx.models.register(run_id, "iris-model")
    model_id, version, problem = signal.args
    assert model_id == model.id and version == "1" and problem == ""
    assert ctx.models.get(model.id).mlflow_version == "1"
