"""Phase 9 pages: Deep Learning builder."""

from __future__ import annotations

import pytest

from app.ui.pages.deep_learning import DeepLearningPage
from tests.fixtures.fake_vision import write_fake_mnist
from tests.ui.test_experiment_service import wait_for


def test_deep_learning_page_edits_and_checks_the_stack(ctx, qtbot) -> None:
    page = DeepLearningPage(ctx)
    qtbot.addWidget(page)
    page.dataset.setCurrentIndex(page.dataset.findData("mnist"))
    assert page.shape_text.text() == "1×28×28 → 10"
    assert page.table.rowCount() == len(page.steps) == 13 and "class LayerStack" in page.code.toPlainText()
    assert not page.run_button.isEnabled() and page.open_button.isEnabled()  # not downloaded yet
    editor = page.editor
    editor.list.setCurrentRow(len(editor.layers) - 1)
    editor.kind.setCurrentIndex(editor.kind.findData("maxpool"))
    for _ in range(3):
        editor.add_selected_kind()  # pooling a vector cannot work
    assert not page.steps and "needs image maps" in page.problem.text() and not page.open_button.isEnabled()
    for _ in range(3):
        editor.remove_selected()
    assert page.steps and not page.problem.isVisibleTo(page)
    editor.list.setCurrentRow(0)
    editor.form_holder.findChild(type(page.epochs), "setting_channels").setValue(8)
    assert editor.layers[0]["channels"] == 8 and page.steps[0].shape == (8, 28, 28)
    editor.move(1)
    assert editor.layers[1]["type"] == "conv" and editor.list.currentRow() == 1
    editor.set_layers([])
    assert [s.kind for s in page.steps] == ["flatten", "linear"]


def test_deep_learning_page_trains_its_network(ctx, qtbot) -> None:
    pytest.importorskip("torch")
    write_fake_mnist(ctx.paths.datasets)
    page = DeepLearningPage(ctx)
    qtbot.addWidget(page)
    page.dataset.setCurrentIndex(page.dataset.findData("mnist"))
    page.refresh()
    assert page.run_button.isEnabled()
    spec = page.spec()
    assert spec.model.model == "layer_stack" and spec.model.params["layers"] == page.editor.layers
    page.epochs.setValue(1)
    run_id = page.train()
    assert run_id and "Queued" in page.train_text.text()
    view = wait_for(qtbot, ctx, run_id, ("completed", "failed"))
    assert view.status == "completed", ctx.experiments.log_lines(run_id, limit=40)
