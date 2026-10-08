"""The Classical ML lab: presets through the queue, CSV import, the builder's scikit-learn
section, and scikit-learn results on the Training and Evaluation pages."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from app.ui.pages.builder import ExperimentBuilderPage  # noqa: E402
from app.ui.pages.classical_ml import ClassicalMlPage  # noqa: E402
from app.ui.pages.evaluation import EvaluationPage  # noqa: E402
from app.ui.pages.training_run import RunDetail  # noqa: E402
from app.ui.widgets.plots import BarList, CurveChart, ScatterChart  # noqa: E402
from core.common.optional import is_installed  # noqa: E402
from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.spec import dump_spec_text  # noqa: E402
from labs.classical_ml.presets import CLASSICAL_PRESETS  # noqa: E402
from tests.ui.test_experiment_service import wait_for  # noqa: E402

PRESETS = {preset.id: preset for preset in CLASSICAL_PRESETS}


def hover(widget, x: float, y: float) -> None:
    widget.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                                      Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                      Qt.KeyboardModifier.NoModifier))


def test_builder_round_trips_every_classical_preset(ctx, qtbot) -> None:
    page = ExperimentBuilderPage(ctx)
    qtbot.addWidget(page)
    for preset in CLASSICAL_PRESETS:
        spec = preset.make()
        if spec.model.model == "xgboost" and not is_installed("xgboost"):
            continue
        page.load(spec)
        assert page.form.tabular and page.form.sklearn.cards.isVisibleTo(page)
        assert not page.form.torch.cards.isVisibleTo(page)
        assert dump_spec_text(page.form.spec()) == dump_spec_text(spec), preset.id
        assert page.problems == [], (preset.id, page.problems)


def test_builder_columns_target_and_yaml_boxes(ctx, qtbot) -> None:
    page = ExperimentBuilderPage(ctx)
    qtbot.addWidget(page)
    page.load(PRESETS["diabetes_forest"].make())
    section = page.form.sklearn
    assert section.features.count() == 10 and section.target.itemText(0).endswith("(target)")
    section.features.item(0).setCheckState(Qt.CheckState.Unchecked)  # age
    section.target.setCurrentIndex(section.target.findData("bmi"))
    page.form.params.setPlainText("n_estimators: 25\nmax_depth: 3")
    page.refresh()
    spec = page.spec
    assert spec is not None and spec.training.target == "bmi"
    assert "age" not in spec.training.features and "bmi" not in spec.training.features
    assert len(spec.training.features) == 8
    assert spec.model.params == {"n_estimators": 25, "max_depth": 3} and page.problems == []
    # Another dataset lists its own columns again, all checked.
    page.form.task.setCurrentIndex(page.form.task.findData(Task.TABULAR_CLASSIFICATION))
    page.form.dataset.select("wine")
    page.form.model.select("random_forest")
    page.refresh()
    assert section.features.count() == 13 and page.spec.training.features is None
    page.form.params.setPlainText("- not a mapping")
    page.refresh()
    assert page.spec is None and "Model options" in page.problems[0] and not page.run_button.isEnabled()
    page.form.params.setPlainText("")
    section.search.setCurrentIndex(section.search.findData("grid"))
    page.refresh()
    assert any("Search space" in p for p in page.problems)
    section.space.setPlainText("n_estimators: [10, 20]")
    page.refresh()
    assert page.spec.training.search.space == {"n_estimators": (10, 20)} and page.problems == []


def test_preset_runs_and_shows_its_results(ctx, qtbot) -> None:
    visited = []
    ctx.navigate = visited.append
    page = ClassicalMlPage(ctx)
    qtbot.addWidget(page)
    assert page.datasets.rowCount() >= 5 and page.models.rowCount() >= 10
    run_button, _edit = page.preset_buttons["iris_logistic"]
    assert run_button.isEnabled()
    run_id = page.run_preset(PRESETS["iris_logistic"])
    assert visited[-1] == "training"
    view = wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000)
    assert view.status == "completed", view.error
    assert view.tabular and ctx.store.latest_metrics(run_id)["test_acc"] > 0.8

    detail = RunDetail(ctx)
    qtbot.addWidget(detail)
    detail.show_run(run_id)
    assert not detail.curves.isVisibleTo(detail) and detail.tabular.isVisibleTo(detail)
    results = detail.tabular
    assert results.scores.rowCount() >= 6 and results.curves_card.isVisibleTo(detail)
    assert results.curve_class.count() == 3 and results.importance.rows()
    assert not results.scatter_card.isVisibleTo(detail)

    evaluation = EvaluationPage(ctx)
    qtbot.addWidget(evaluation)
    assert evaluation.run_id == run_id and evaluation.checkpoint_combo.currentData() == "model"
    assert evaluation.split_combo.count() == 1 and evaluation.matrix.matrix.sum() == 30
    assert evaluation.source_combo.currentText().startswith("End of fitting")
    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=120_000):
        assert evaluation.evaluate()
    qtbot.waitUntil(lambda: evaluation.source_combo.count() == 2, timeout=5_000)
    assert evaluation.report["checkpoint"] == "model.joblib"


def test_regression_and_clustering_results(ctx, qtbot) -> None:
    detail = RunDetail(ctx)
    qtbot.addWidget(detail)
    for key in ("diabetes_forest", "iris_kmeans"):
        run_id = ctx.experiments.launch(PRESETS[key].make())
        assert wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000).status == "completed"
        detail.show_run(run_id)
        results = detail.tabular
        assert results.scatter_card.isVisibleTo(detail) and not results.curves_card.isVisibleTo(detail)
    assert results.scatter_card.title.text() == "Clusters" and len(results.scatter.groups) == 3
    assert "blue: cluster 0" in results.legend.text()
    evaluation = EvaluationPage(ctx)
    qtbot.addWidget(evaluation)
    evaluation.run_combo.setCurrentIndex(evaluation.run_combo.findData(run_id))
    assert evaluation.tabular.isVisibleTo(evaluation)
    assert not evaluation.evaluate_button.isVisibleTo(evaluation)
    assert "nothing is held out" in evaluation.no_rescore.text()


def test_csv_import_from_the_page(ctx, qtbot, tmp_path) -> None:
    rng = np.random.default_rng(1)
    lines = ["width,height,shape"] + [f"{w:.2f},{h:.2f},{'tall' if h > w else 'wide'}"
                                      for w, h in rng.uniform(1, 10, size=(80, 2))]
    source = tmp_path / "shapes.csv"
    source.write_text("\n".join(lines), encoding="utf-8")
    page = ClassicalMlPage(ctx)
    qtbot.addWidget(page)
    builder = ExperimentBuilderPage(ctx)
    qtbot.addWidget(builder)
    importer = page.importer
    assert not importer.import_button.isEnabled()
    assert not importer.load(tmp_path / "missing.csv") and "could not be read" in importer.file_text.text()
    assert importer.load(source) and importer.columns.rowCount() == 3
    assert importer.target.currentData() == "shape"
    assert importer.task.currentData() == "tabular_classification"
    rows = page.datasets.rowCount()
    with qtbot.waitSignal(ctx.downloads.imported, timeout=30_000):
        importer.start_import()
    assert importer.state.text() == "Imported" and "csv-shapes" in importer.message.text()
    assert page.datasets.rowCount() == rows + 1 and ctx.catalog.datasets.status("csv-shapes") == "ready"
    builder.form.task.setCurrentIndex(builder.form.task.findData(Task.TABULAR_CLASSIFICATION))
    builder.form.dataset.select("csv-shapes")
    builder.refresh()
    assert builder.form.dataset.card_id == "csv-shapes" and builder.form.sklearn.features.count() == 2
    assert builder.problems == []


def test_plot_widgets(qtbot) -> None:
    curve = CurveChart("ROC", "fpr", "tpr", chance=True)
    qtbot.addWidget(curve)
    curve.resize(300, 240)
    curve.set_curve([[0, 0], [0.1, 0.8], [1, 1]], "AUC 0.95")
    hover(curve, curve.LEFT + 0.1 * (300 - curve.LEFT - curve.RIGHT), 100)
    assert curve.hover_text() == "fpr 0.100 · tpr 0.800"
    scatter = ScatterChart("", "x", "y")
    qtbot.addWidget(scatter)
    scatter.resize(300, 240)
    scatter.set_points([[0, 0, 0], [1, 1, 4], [2, 0, 1]], [f"g{i}" for i in range(5)])
    assert scatter.colour_of(0) != scatter.colour_of(4)  # one group coloured, the rest folded
    assert scatter.colour_of(1) == scatter.colour_of(4)
    hover(scatter, scatter.LEFT + (300 - scatter.LEFT - scatter.RIGHT) / 2, scatter.TOP)  # point (1, 1)
    assert scatter.hover_text() == "1, 1 · g4"
    bars = BarList()
    qtbot.addWidget(bars)
    bars.set_rows([["a", 0.5, 0.1], ["b", 0.25, 0.0]])
    assert bars.height() == 2 * BarList.ROW + 4 and bars.rows()[0] == ("a", 0.5, 0.1)
