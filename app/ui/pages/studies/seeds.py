"""One seed study: every metric's mean, spread and 95 % interval over the seeds, each run, and a
comparison with another study (Welch's t-test, with the compute budgets checked first)."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from ....services.context import AppContext
from ...widgets import Card, DataTable, ResponsiveRow, label

__all__ = ["SeedView", "interval_text"]


def interval_text(s: dict[str, Any]) -> str:
    return "—" if s.get("ci_low") is None else f"{s['ci_low']:.4f} – {s['ci_high']:.4f}"


class SeedView(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.group_id: str | None = None
        self.result: dict[str, Any] | None = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        stats = Card("Over the seeds", "finished runs only", padded=False)
        self.stats = DataTable(("Metric", "n", "Mean", "SD", "95 % interval", "Min", "Max"),
                               mono_columns=(1, 2, 3, 4, 5, 6), stretch_column=0)
        stats.add(self.stats)
        self.method = label("", "CardCaption", wrap=True)
        stats.add(self.method)
        column.addWidget(stats)
        runs = Card("Runs", padded=False)
        self.runs = DataTable(("Run", "Status", "Seed", "Score"), mono_columns=(0, 2, 3), stretch_column=0)
        runs.add(self.runs)
        versus = Card("Compare with another study", "Welch's t-test; budgets must match")
        row = QHBoxLayout()
        self.other = QComboBox()
        self.other.setMinimumWidth(200)
        self.score_box = QComboBox()
        self.score_box.setMinimumWidth(140)
        for box in (self.other, self.score_box):
            box.currentIndexChanged.connect(lambda _: self._compare())
            row.addWidget(box)
        row.addStretch(1)
        versus.add(row)
        self.verdict = label("", "Body", wrap=True)
        versus.add(self.verdict)
        column.addWidget(ResponsiveRow([(runs, 1), (versus, 1)], breakpoint=900))

    def set_group(self, group_id: str | None) -> None:
        self.group_id = group_id
        self.stats.clear_rows()
        self.runs.clear_rows()
        if group_id is None:
            return
        service = self.ctx.studies
        summary = service.summary(group_id)
        for name, s in summary.items():
            if not s.get("n"):
                continue
            sd = "—" if s.get("sd") is None else f"{s['sd']:.4f}"
            self.stats.add_row((name, str(s["n"]), f"{s['mean']:.4f}", sd, interval_text(s),
                                f"{s['min']:.4f}", f"{s['max']:.4f}"))
        first = next((s for s in summary.values() if s.get("n")), None)
        self.method.setText(f"Method: {first['method']}." if first else "No run has finished yet.")
        rows = service.rows(group_id)
        names = service.metric_names(rows)
        key = names[0] if names else None
        for row in rows:
            value = row["metrics"].get(key) if key else None
            self.runs.add_row((row["run_id"][:8], row["status"], str(row["seed"]),
                               "—" if value is None else f"{value:.4f}"))
        self.score_box.blockSignals(True)
        self.other.blockSignals(True)
        self.score_box.clear()
        for name in names:
            self.score_box.addItem(name, name)
        self.other.clear()
        self.other.addItem("Choose a study", None)
        for group in service.groups("seeds"):
            if group["id"] != group_id:
                self.other.addItem(f"{group['name']} · {group['id'][:6]}", group["id"])
        self.score_box.blockSignals(False)
        self.other.blockSignals(False)
        self._compare()

    def _compare(self) -> None:
        other, metric = self.other.currentData(), self.score_box.currentData()
        if self.group_id is None or other is None or metric is None:
            self.result = None
            self.verdict.setText("Choose a second study to test whether the difference is more than seed "
                                 "noise.")
            return
        self.result = result = self.ctx.studies.compare(other, self.group_id, metric)
        if result.get("p_value") is None:
            self.verdict.setText(result["method"])
            return
        lines = [f"{metric}: this study minus the other = {result['difference']:+.4f} "
                 f"(95 % interval {result['ci_low']:+.4f} … {result['ci_high']:+.4f}), "
                 f"t = {result['t']:.2f}, df = {result['df']:.1f}, p = {result['p_value']:.3g}."]
        if not result["matched_budget"]:
            lines.append("The compute budgets differ, so this is not a like-for-like comparison: "
                         + "; ".join(result["budget_differences"]) + ".")
        lines.append(result["method"] + ".")
        self.verdict.setText("\n".join(lines))
