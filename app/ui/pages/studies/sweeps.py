"""One sweep, the way an experiment tracker shows it: the runs with their parameters, state,
runtime and score; parallel coordinates from the parameters to the score; which parameters matter
for the score (importance) and which way they push it (correlation); and the validation curves of
the best runs overlaid."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from ....services.context import AppContext
from ...widgets import Card, DataTable, ResponsiveRow, label
from ...widgets.overlay import OverlayChart
from ...widgets.parallel import Axis, ParallelCoordinates
from ...widgets.plots import BarList

__all__ = ["SweepView"]

#: Curves overlaid for the best runs (at most eight, never cycled).
CURVES = (("val_acc", "Validation accuracy", True), ("val_loss", "Validation loss", False))


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "—" if value is None else str(value)


class SweepView(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.group_id: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.ranking: list[dict[str, Any]] = []
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(label("Score", "CardCaption"))
        self.score_box = QComboBox()
        self.score_box.setMinimumWidth(160)
        self.score_box.currentIndexChanged.connect(lambda _: self._draw())
        head.addWidget(self.score_box)
        head.addStretch(1)
        column.addLayout(head)
        parallel = Card("Parameters → score", "each line is a run")
        self.parallel = ParallelCoordinates()
        parallel.add(self.parallel)
        column.addWidget(parallel)
        importance = Card("Parameter importance", "for the chosen score")
        self.importance = BarList()
        importance.add(self.importance)
        self.correlation = DataTable(("Parameter", "Importance", "Correlation"), mono_columns=(1, 2),
                                     stretch_column=0)
        importance.add(self.correlation)
        importance.add(label("Importance: a random forest fitted from the parameters to the score "
                             "(impurity importance, summing to 1). Correlation: Pearson, numeric parameters "
                             "only; + means a larger value goes with a higher score.", "CardCaption",
                             wrap=True))
        curves = Card("Best runs", "validation curves of the eight best")
        self.curve = QComboBox()
        for key, title, _ in CURVES:
            self.curve.addItem(title, key)
        self.curve.currentIndexChanged.connect(lambda _: self._draw_curves())
        curves.add_head_widget(self.curve)
        self.chart = OverlayChart()
        curves.add(self.chart)
        column.addWidget(ResponsiveRow([(importance, 1), (curves, 1)], breakpoint=900))
        runs = Card("Runs", padded=False)
        self.table = DataTable(("Run", "State", "Runtime s", "Score"), mono_columns=(0, 2, 3),
                               stretch_column=0)
        runs.add(self.table)
        column.addWidget(runs)

    def set_group(self, group_id: str | None) -> None:
        self.group_id = group_id
        self.rows = self.ctx.studies.rows(group_id) if group_id else []
        names = self.ctx.studies.metric_names(self.rows)
        chosen = self.score_box.currentData()
        self.score_box.blockSignals(True)
        self.score_box.clear()
        for name in names:
            self.score_box.addItem(name, name)
        self.score_box.setCurrentIndex(max(self.score_box.findData(chosen), 0))
        self.score_box.blockSignals(False)
        self._draw()

    def _draw(self) -> None:
        metric = self.score_box.currentData()
        params = list(self.ctx.studies.group(self.group_id).get("space", {})) if self.group_id else []
        headers = ("Run", "State", *[p.split(".")[-1] for p in params], "Runtime s", metric or "Score")
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(list(headers))
        self.table.clear_rows()
        ordered = sorted(self.rows, key=lambda r: -(r["metrics"].get(metric) or float("-inf"))) if metric \
            else self.rows
        for row in ordered:
            runtime = "—" if row["duration_s"] is None else f"{row['duration_s']:.1f}"
            self.table.add_row((row["run_id"][:8], row["status"], *[_value(row["params"][p]) for p in params],
                                runtime, _value(row["metrics"].get(metric))))
        done = [r for r in self.rows if metric and metric in r["metrics"]]
        axes = [Axis(p.split(".")[-1], [r["params"][p] for r in done]) for p in params]
        if metric:
            axes.append(Axis(metric, [r["metrics"][metric] for r in done]))
        self.parallel.set_data(axes, [r["run_id"][:8] for r in done])
        self.ranking = self.ctx.studies.importance(self.group_id, metric) if self.group_id and metric else []
        self.importance.set_rows([(r["parameter"].split(".")[-1], r["importance"]) for r in self.ranking])
        self.correlation.clear_rows()
        for r in self.ranking:
            correlation = "—" if r["correlation"] is None else f"{r['correlation']:+.2f}"
            self.correlation.add_row((r["parameter"], f"{r['importance']:.3f}", correlation))
        self._draw_curves()

    def _draw_curves(self) -> None:
        metric, key = self.score_box.currentData(), self.curve.currentData()
        best = sorted((r for r in self.rows if metric in r["metrics"]),
                      key=lambda r: -r["metrics"][metric])[:8] if metric else []
        series = [(r["run_id"][:8], self.ctx.store.metric_history(r["run_id"], key)) for r in best]
        series = [s for s in series if s[1]]
        percent = dict((k, p) for k, _, p in CURVES)[key]
        self.chart.set_series(self.curve.currentText(), series, percent=percent)
