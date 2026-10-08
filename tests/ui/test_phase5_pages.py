"""Evaluation, reproduction and the MLflow page."""

from __future__ import annotations

import pytest

from app.services.mlflow_ui import MlflowUi, free_port
from app.ui.pages.evaluation import EvaluationPage
from app.ui.pages.mlflow_page import MlflowPage
from core.common import AppConfig
from tests.fixtures.fake_vision import write_fake_mnist


def finished_run(ctx, qtbot) -> str:
    pytest.importorskip("torch")
    from tests.ui.test_experiment_service import spec, wait_for
    write_fake_mnist(ctx.paths.datasets)
    run_id = ctx.experiments.launch(spec(epochs=1, steps=2))
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed")).status == "completed"
    return run_id


def test_evaluation_page_and_re_evaluation(ctx, qtbot) -> None:
    empty = EvaluationPage(ctx)
    qtbot.addWidget(empty)
    assert empty.empty.isVisibleTo(empty) and not empty.evaluate_button.isEnabled()
    run_id = finished_run(ctx, qtbot)
    page = EvaluationPage(ctx)
    qtbot.addWidget(page)
    assert page.run_id == run_id and page.source_combo.count() == 1
    assert page.tiles["acc"].value.text().endswith("%") and page.table.rowCount() == 10
    assert len(page.matrix.classes) == 10 and page.matrix.matrix.sum() == 60
    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=120_000):
        assert page.evaluate()
    qtbot.waitUntil(lambda: page.source_combo.count() == 2, timeout=5_000)
    assert page.source_combo.currentText().startswith("Re-evaluated · test split · best")
    assert page.state.text() == "Done" and page.report["split"] == "test"
    page.split_combo.setCurrentIndex(1)
    page.checkpoint_combo.setCurrentIndex(1)
    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=120_000):
        page.evaluate()
    qtbot.waitUntil(lambda: page.source_combo.count() == 3, timeout=5_000)
    assert page.report["split"] == "val" and page.report["checkpoint"] == "last.pt"
    assert ctx.store.run(run_id)["status"] == "completed"  # evaluation leaves the run alone


def test_reproduce_links_and_compares(ctx, qtbot) -> None:
    from app.ui.pages.training_run import RunDetail
    from tests.ui.test_experiment_service import wait_for

    original = finished_run(ctx, qtbot)
    notes = ctx.experiments.reproduction_notes(original)
    assert all(isinstance(note, str) for note in notes)
    child = ctx.experiments.reproduce(original)
    view = ctx.experiments.view(child)
    assert view.parent_run_id == original
    assert view.experiment_id == ctx.experiments.view(original).experiment_id
    assert (view.run_dir / "experiment.yaml").read_text() == (
        ctx.experiments.view(original).run_dir / "experiment.yaml").read_text()
    assert wait_for(qtbot, ctx, child, ("completed", "failed")).status == "completed"
    detail = RunDetail(ctx)
    qtbot.addWidget(detail)
    detail.show_run(child)
    text = detail.lineage.text()
    assert text.startswith(f"Reproduction of run {original}") and "test_acc:" in text
    assert detail.values.value_text("MLflow run") == "not tracked"  # switched off in tests
    snapshot = (view.run_dir / "snapshot.json")
    assert snapshot.is_file()


def test_mlflow_page_lists_runs(ctx, qtbot) -> None:
    page = MlflowPage(ctx)
    qtbot.addWidget(page)
    page.show_runs([{"experiment": "mnist", "run": "abc123", "lab_run": "r1", "status": "FINISHED",
                     "started": 1_700_000_000_000, "test_acc": 0.98, "model": "tiny_vgg",
                     "dataset": "mnist"}])
    assert page.table.rowCount() == 1 and page.table.item(0, 4).text() == "0.9800"
    page.show_runs([], "boom")
    assert "Could not read MLflow: boom" in page.message.text()


def test_mlflow_ui_reports_a_failed_start(ctx, qtbot, tmp_path) -> None:
    ui = MlflowUi(ctx.paths, AppConfig(python_executable=str(tmp_path / "no-python")))
    command = ui.command(5123)
    assert "--backend-store-uri" in command and "127.0.0.1" in command and "5123" in command
    with qtbot.waitSignal(ui.state_changed, timeout=10_000, check_params_cb=lambda s: s == "failed"):
        ui.start()
    assert ui.state == "failed" and ui.url is None and "could not start" in ui.output
    ui.stop()
    assert 5000 <= free_port() < 5050
