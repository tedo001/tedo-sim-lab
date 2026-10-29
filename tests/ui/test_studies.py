"""Seed studies and sweeps end to end: queued from the builder's card, finished by real workers,
summarised (mean, spread, t-interval), compared (Welch, budgets) and drawn on Studies & Sweeps."""

from __future__ import annotations

import csv
import json

import pytest

from core.experiment_engine import studies
from core.experiment_engine.stats import summarize, welch
from tests.ui.test_experiment_service import wait_for

pytest.importorskip("sklearn")


def _finish_all(ctx, qtbot, run_ids):
    for run_id in run_ids:
        assert wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000).status == "completed"


def _iris():
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    return CLASSICAL_PRESETS[0].make()


def test_summary_and_welch_match_scipy():
    from scipy import stats

    a, b = [0.90, 0.92, 0.91, 0.95, 0.93], [0.94, 0.96, 0.95, 0.97, 0.95]
    s = summarize(a)
    low, high = stats.t.interval(0.95, len(a) - 1, loc=s["mean"], scale=stats.sem(a))
    assert (s["ci_low"], s["ci_high"]) == pytest.approx((low, high))
    w = welch(a, b)
    reference = stats.ttest_ind(b, a, equal_var=False)
    assert w["t"] == pytest.approx(reference.statistic) and w["p_value"] == pytest.approx(reference.pvalue)
    assert summarize([1.0])["ci_low"] is None and "one value" in summarize([1.0])["method"]


def test_sweep_and_study_specs():
    spec = _iris()
    space = {"model.params.C": [0.1, 1.0, 10.0], "data.split.stratify": [True, False]}
    grid = studies.sweep(spec, space, "s1")
    assert len(grid) == 6 and all("sweep:s1" in s.tags for s in grid)
    assert sorted({s.model.params["C"] for s in grid}) == [0.1, 1.0, 10.0]
    assert len(studies.sweep(spec, {"model.params.C": [0.1, 1.0, 10.0]}, "s2", method="random", count=2)) == 2
    with pytest.raises(ValueError):
        studies.sweep(spec, {"seed": "not a list"}, "s3")
    seeds = studies.seed_study(spec, [0, 1, 2], "st")
    assert [s.seed for s in seeds] == [0, 1, 2] and "seed:2" in seeds[2].tags
    first = studies.budget(seeds[0], {"train_rows": 120})
    second = studies.budget(seeds[1], {"train_rows": 60})
    assert studies.budget_mismatch(first, first) == []
    assert "samples seen" in studies.budget_mismatch(first, second)[0]


def _pages(ctx, qtbot):
    from app.ui.pages.builder import ExperimentBuilderPage
    from app.ui.pages.studies import StudiesPage

    studies_page, builder = StudiesPage(ctx), ExperimentBuilderPage(ctx)
    qtbot.addWidget(studies_page)
    qtbot.addWidget(builder)
    ctx.show_page = lambda page_id: studies_page if page_id == "studies" else builder
    return builder, studies_page


def test_seed_study_from_the_builder(ctx, qtbot):
    builder, page = _pages(ctx, qtbot)
    builder.load(_iris())
    builder.studies.seeds.setText("0-2")
    study_id = builder.studies.queue_study()
    assert study_id, builder.studies.note.text()
    runs = ctx.studies.group(study_id)["runs"]
    assert len(runs) == 3
    _finish_all(ctx, qtbot, runs)
    summary = ctx.studies.summary(study_id)
    assert summary["test_acc"]["n"] == 3 and summary["test_acc"]["ci_low"] <= summary["test_acc"]["mean"]
    page.show_group(study_id)
    assert page.kind == "seeds" and page.views["seeds"].stats.rowCount() >= 1
    written = page.export()
    data = json.loads(written["json"].read_text())
    assert len(data["runs"]) == 3 and data["statistics"]["test_acc"]["method"].startswith("mean ± t")
    assert {row["seed"] for row in csv.DictReader(written["csv"].open())} == {"0", "1", "2"}
    assert "95 % interval" in written["md"].read_text()
    # A second study with another budget: Welch's test, and the budget difference is named.
    other = _iris().model_copy(update={"data": _iris().data.model_copy(update={"subset": 60})})
    second = ctx.studies.launch_study(other, [0, 1, 2])
    _finish_all(ctx, qtbot, ctx.studies.group(second)["runs"])
    result = ctx.studies.compare(study_id, second, "test_acc")
    assert result["matched_budget"] is False and any("subset" in d for d in result["budget_differences"])
    page.show_group(second)
    view = page.views["seeds"]
    view.other.setCurrentIndex(view.other.findData(study_id))
    assert "compute budgets differ" in view.verdict.text() or "Welch" in view.verdict.text()


def test_sweep_page(ctx, qtbot):
    _builder, page = _pages(ctx, qtbot)
    spec = _iris()
    sweep_id = ctx.studies.launch_sweep(spec, {"model.params.C": [0.01, 0.1, 1.0, 10.0]})
    _finish_all(ctx, qtbot, ctx.studies.group(sweep_id)["runs"])
    ranking = ctx.studies.importance(sweep_id, "test_acc")
    assert ranking[0]["parameter"] == "model.params.C" and ranking[0]["importance"] == pytest.approx(1.0)
    page.show_group(sweep_id)
    view = page.views["sweep"]
    assert page.kind == "sweep" and view.table.rowCount() == 4
    assert len(view.parallel.axes) == 2 and len(view.parallel.labels) == 4
    assert view.correlation.rowCount() == 1
