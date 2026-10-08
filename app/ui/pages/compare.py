"""Compare Experiments: up to eight finished runs side by side (scores, size, time, device, peak
GPU memory, measured latency), one metric's curves overlaid, and the comparison exported as CSV,
JSON, Markdown or PDF into the workspace's reports/ folder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from ...services.context import AppContext
from ...services.reports import comparison_row, export_report, markdown_by_measure, used_columns
from ..icons import icon
from ..theme.tokens import SERIES
from ..widgets import Card, DataTable, Page, label
from ..widgets.overlay import OverlayChart

__all__ = ["ComparePage"]

#: Curves the comparison can overlay: (metric, title, a 0-1 share).
CURVES = (("val_acc", "Validation accuracy", True), ("val_loss", "Validation loss", False),
          ("train_acc", "Training accuracy", True), ("train_loss", "Training loss", False),
          ("val_f1", "Validation F1", True), ("lr", "Learning rate", False))


def _swatch_icon(colour: str) -> QIcon:
    """A small square in a series colour, for a run's column header."""
    pixmap = QPixmap(10, 10)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


class ComparePage(Page):
    MAX = len(SERIES)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Compare Experiments", "runs side by side, curves overlaid, exported as a report",
                         parent)
        self.ctx = ctx
        self.last_export: dict[str, Path] = {}

        pick = Card("Runs", f"tick up to {self.MAX} finished runs")
        self.runs = QListWidget()
        self.runs.setMaximumHeight(200)
        self.runs.itemChanged.connect(self._ticked)
        pick.add(self.runs)
        self.limit = label("", "CardCaption", wrap=True)
        pick.add(self.limit)
        self.body.addWidget(pick)

        self.table_card = Card("Side by side", padded=False)
        self.table = DataTable(("Run",), stretch_column=0)
        self.table_card.add(self.table)
        self.body.addWidget(self.table_card)

        curves = Card("Curves", "one metric, every ticked run that has per-epoch values")
        self.metric = QComboBox()
        for key, title, _share in CURVES:
            self.metric.addItem(title, key)
        self.metric.currentIndexChanged.connect(lambda _: self._draw())
        curves.add_head_widget(self.metric)
        self.chart = OverlayChart()
        curves.add(self.chart)
        self.curves_card = curves
        self.body.addWidget(curves)

        export = Card("Export", "into this workspace's reports/ folder")
        row = QHBoxLayout()
        choices = (("CSV", ("csv",)), ("JSON", ("json",)), ("Markdown", ("md",)), ("PDF", ("pdf",)),
                   ("All four", ("csv", "json", "md", "pdf")))
        for title, formats in choices:
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, f=formats: self.export(f))
            row.addWidget(button)
        self.open_button = QPushButton(icon("folder-open"), "Open folder")
        self.open_button.clicked.connect(self._open_folder)
        self.open_button.setEnabled(False)
        row.addWidget(self.open_button)
        row.addStretch(1)
        export.add(row)
        self.export_text = label("", "Mono", wrap=True)
        export.add(self.export_text)
        self.export_card = export
        self.body.addWidget(export)
        self.body.addStretch(1)
        ctx.experiments.run_changed.connect(lambda _: self._fill_runs())
        ctx.benchmarks.changed.connect(self.refresh)
        self._fill_runs()

    # Choosing ---------------------------------------------------------------------------
    def _fill_runs(self) -> None:
        ticked = set(self.selected_ids())
        self.runs.blockSignals(True)
        self.runs.clear()
        for view in self.ctx.experiments.views(limit=200):
            if not view.finished_ok:
                continue
            item = QListWidgetItem(f"{view.name} · {view.dataset} · {view.model} · {view.id[:8]}")
            item.setData(Qt.ItemDataRole.UserRole, view.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if view.id in ticked else Qt.CheckState.Unchecked)
            self.runs.addItem(item)
        self.runs.blockSignals(False)
        self.refresh()

    def selected_ids(self) -> list[str]:
        items = [self.runs.item(i) for i in range(self.runs.count())]
        return [item.data(Qt.ItemDataRole.UserRole) for item in items
                if item.checkState() == Qt.CheckState.Checked]

    def select(self, run_ids: list[str]) -> None:
        self.runs.blockSignals(True)
        for i in range(self.runs.count()):
            item = self.runs.item(i)
            wanted = item.data(Qt.ItemDataRole.UserRole) in run_ids[: self.MAX]
            item.setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
        self.runs.blockSignals(False)
        self.refresh()

    def _ticked(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.CheckState.Checked and len(self.selected_ids()) > self.MAX:
            self.runs.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.runs.blockSignals(False)
            self.limit.setText(f"Compare up to {self.MAX} runs at a time: each gets its own colour, and "
                               "colours are never reused.")
            return
        self.limit.setText("")
        self.refresh()

    # Showing ----------------------------------------------------------------------------
    def rows(self) -> list[dict]:
        return [comparison_row(self.ctx.experiments.view(run_id), self.ctx.store, self.ctx.benchmarks)
                for run_id in self.selected_ids()]

    def refresh(self) -> None:
        """Rebuild the table: one row per measure, one column per run (a colour swatch in its
        header matches its curve), so up to eight runs fit side by side."""
        rows = self.rows()
        measures = [c for c in used_columns(rows) if c.key not in ("name", "run_id")]
        old = self.table
        self.table = DataTable(("Measure", *(row["name"] for row in rows)),
                               mono_columns=tuple(range(1, len(rows) + 1)), stretch_column=0)
        old.hide()
        old.setParent(None)
        old.deleteLater()
        for index, row in enumerate(rows):
            header = self.table.horizontalHeaderItem(index + 1)
            header.setIcon(_swatch_icon(SERIES[index]))
            header.setToolTip(f"{row['name']} · run {row['run_id']}")
        self.table_card.add(self.table)
        for column in measures:
            self.table.add_row([column.title, *(column.show(row[column.key]) if column.key in row else "—"
                                                 for row in rows)])
        self.table.add_row(["Run id", *(row["run_id"][:8] for row in rows)])
        self.runs.setMaximumHeight(min(200, 26 * max(self.runs.count(), 1) + 8))
        for card in (self.table_card, self.curves_card, self.export_card):
            card.setVisible(bool(rows))
        self._draw()

    def _draw(self) -> None:
        key = self.metric.currentData()
        share = next(s for k, _t, s in CURVES if k == key)
        series = []
        for run_id in self.selected_ids():
            view = self.ctx.experiments.view(run_id)
            series.append((view.name, [] if view.tabular else self.ctx.store.metric_history(run_id, key)))
        self.chart.set_series(self.metric.currentText(), series, percent=share)

    # Exporting --------------------------------------------------------------------------
    def markdown(self, rows: list[dict], with_chart: bool) -> str:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Comparison of {len(rows)} runs", "",
                 f"TEDO AI Research Lab · {stamp} · workspace `{self.ctx.paths.workspace}`", "",
                 markdown_by_measure(rows, used_columns(rows)), ""]
        if with_chart:
            lines += [f"## {self.metric.currentText()}", "", "![curves](curves.png)", ""]
        lines += ["## Runs", ""]
        for row in rows:
            lines.append(f"- **{row['name']}** (run `{row['run_id']}`): {row['dataset']} · {row['model']} · "
                         f"{row['status']} on {row['device'] or 'unknown device'}"
                         + (f" · commit `{row['git_commit'][:12]}`" if row.get("git_commit") else "")
                         + (f" · MLflow run `{row['mlflow_run_id']}`" if row.get("mlflow_run_id") else ""))
        return "\n".join(lines) + "\n"

    def export(self, formats: tuple[str, ...]) -> dict[str, Path]:
        rows = self.rows()
        if not rows:
            return {}
        with_chart = any(points for _name, points in self.chart.series)
        images = {}
        if with_chart:  # drawn again at page width, without hover marks
            printable = OverlayChart()
            printable.set_series(self.chart.title, self.chart.series, percent=self.chart.percent)
            printable.resize(760, 340)
            images["curves.png"] = printable.grab().toImage()
            printable.deleteLater()
        self.last_export = export_report(self.ctx.paths.reports, f"compare-{len(rows)}-runs",
                                         self.markdown(rows, with_chart), rows=rows, images=images,
                                         formats=formats, extra={"metric": self.metric.currentData()})
        self.export_text.setText("\n".join(str(path) for path in self.last_export.values()))
        self.open_button.setEnabled(bool(self.last_export))
        return self.last_export

    def _open_folder(self) -> None:
        if self.last_export:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(next(iter(self.last_export.values())).parent)))
