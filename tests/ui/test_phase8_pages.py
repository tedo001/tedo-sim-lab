"""Benchmarking, Compare Experiments, run reports and the overlaid-curves chart."""

from __future__ import annotations

import json

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from app.ui.pages.benchmarking import BenchmarkingPage, parse_batches
from app.ui.pages.compare import ComparePage
from app.ui.widgets.overlay import OverlayChart
from tests.ui.test_experiment_service import wait_for


def _finished(ctx, qtbot, preset_index: int) -> str:
    pytest.importorskip("sklearn")
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    run_id = ctx.experiments.launch(CLASSICAL_PRESETS[preset_index].make())
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000).status == "completed"
    return run_id


def test_parse_batches() -> None:
    assert parse_batches("64, 1 8,8") == [1, 8, 64]
    for bad in ("", "0", "two", "1.5"):
        with pytest.raises(ValueError):
            parse_batches(bad)


def test_benchmarking_page_runs_an_untrained_model(ctx, qtbot) -> None:
    pytest.importorskip("torch")
    page = BenchmarkingPage(ctx)
    qtbot.addWidget(page)
    assert page.empty.isVisibleTo(page) and not page.detail.isVisibleTo(page)
    page.source.setCurrentIndex(page.source.findData("card:simple_cnn"))
    assert page.shape_row.isEnabled() and not page.precision.isEnabled()  # fp16 needs CUDA
    page.channels.setValue(1)
    page.size.setValue(28)
    page.batches.setText("1, 4")
    page.warmup.setValue(1)
    page.iterations.setValue(3)
    request, title = page.request()
    assert request["model"] == "simple_cnn" and request["input_shape"] == [1, 28, 28] and "untrained" in title
    with qtbot.waitSignal(ctx.benchmarks.finished, timeout=180_000) as signal:
        bench_id = page.run()
    assert signal.args == [bench_id, "completed"]
    assert page.selected == bench_id and page.rows.rowCount() == 2
    assert page.state.text() == "Done" and len(page.throughput_bars.rows()) == 2
    page.batches.setText("zero")
    assert page.run() is None and "whole numbers" in page.message.text()


def test_compare_and_export(ctx, qtbot) -> None:
    first, second = _finished(ctx, qtbot, 0), _finished(ctx, qtbot, 2)  # iris logistic, wine SVM
    page = ComparePage(ctx)
    qtbot.addWidget(page)
    assert page.runs.count() == 2 and not page.table_card.isVisibleTo(page)
    page.select([first, second])
    assert page.selected_ids() == [second, first] or set(page.selected_ids()) == {first, second}
    assert page.table.columnCount() == 3 and page.table_card.isVisibleTo(page)  # measure + two runs
    measures = [page.table.item(row, 0).text() for row in range(page.table.rowCount())]
    assert "Test accuracy" in measures and "Test R²" not in measures  # only what some run has
    assert not page.table.horizontalHeaderItem(1).icon().isNull()  # the run's colour
    assert all(not points for _name, points in page.chart.series)  # scikit-learn: no per-epoch curves
    written = page.export(("csv", "json", "md", "pdf"))
    assert set(written) == {"csv", "json", "md", "pdf"}
    assert written["pdf"].read_bytes()[:4] == b"%PDF" and written["pdf"].stat().st_size > 1000
    assert "| Measure |" in written["md"].read_text() and written["csv"].read_text().startswith("name,")
    assert len(json.loads(written["json"].read_text())["rows"]) == 2
    assert written["md"].parent.parent == ctx.paths.reports and page.open_button.isEnabled()


def test_compare_limits_runs_to_the_palette(ctx, qtbot) -> None:
    page = ComparePage(ctx)
    qtbot.addWidget(page)
    assert page.MAX == 8
    chart = OverlayChart()
    with pytest.raises(ValueError, match="at most 8"):
        chart.set_series("x", [(str(i), [(1, 0.5)]) for i in range(9)])


def test_overlay_chart_hover_lists_every_run(qtbot) -> None:
    chart = OverlayChart()
    qtbot.addWidget(chart)
    chart.resize(500, 300)
    chart.set_series("Validation accuracy", [("a", [(1, 0.5), (2, 0.7)]), ("b", [(1, 0.4), (2, 0.65)])],
                     percent=True)
    area = chart._area()
    x = area.right()
    chart.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(x, 100), QPointF(x, 100),
                                     Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                     Qt.KeyboardModifier.NoModifier))
    assert chart.hover_lines() == ["epoch 2", "a: 70.00%", "b: 65.00%"]


def test_run_report(ctx, qtbot) -> None:
    from app.ui.pages.training_run import RunDetail

    run_id = _finished(ctx, qtbot, 0)
    detail = RunDetail(ctx)
    qtbot.addWidget(detail)
    detail.show_run(run_id)
    assert detail.report_button.isVisibleTo(detail)
    written = detail.export_report()
    text = written["md"].read_text()
    assert text.startswith("# iris-logistic-regression") and "```yaml" in text and "Test accuracy" in text
    assert "Python" in text and written["pdf"].read_bytes()[:4] == b"%PDF"
    assert "Report written to" in detail.registered.text()
