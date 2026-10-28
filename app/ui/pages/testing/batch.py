"""Batch tests: a trained run's answers for every image or WAV file in a folder, or every row of a CSV
file (texts from one column; table and quantum runs from the feature columns). With a label column
(or class-named sub-folders) the batch is scored; the answers export as CSV into reports/."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ....services.context import AppContext
from ...icons import icon
from ...widgets import DataTable, PathLabel, label

__all__ = ["BatchTest", "collect"]

IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
LIMIT = 2000


def collect(kind: str, source: Path, column: str | None, features: list[str]) -> tuple[list[Any], list[str],
                                                                                       list[str]]:
    """(inputs, names, true labels or "") for ``kind`` from a folder or a CSV file."""
    if source.is_dir():
        suffixes = {".wav"} if kind == "audio" else IMAGES
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in suffixes)[:LIMIT]
        inputs = [str(p) if kind == "audio" else {"path": str(p)} for p in files]
        truth = [p.parent.name if p.parent != source else "" for p in files]
        return inputs, [str(p.relative_to(source)) for p in files], truth
    with source.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))[:LIMIT]
    if kind == "text":
        if not column:
            raise ValueError("choose the column that holds the texts")
        inputs = [row.get(column, "") for row in rows]
    else:
        missing = [name for name in features if rows and name not in rows[0]]
        if missing:
            raise ValueError(f"the CSV file has no column {', '.join(missing)}")
        inputs = [{name: _number(row.get(name)) for name in features} for row in rows]
    truth = [row.get("label") or row.get("target") or row.get("class") or "" for row in rows]
    return inputs, [f"row {i + 1}" for i in range(len(rows))], truth


def _number(text: Any) -> Any:
    if text in (None, ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


class BatchTest(QWidget):
    def __init__(self, ctx: AppContext, send: Callable[[str, dict, Callable, Callable, Callable], str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx, self.send = ctx, send
        self.run_dir: Path | None = None
        self.kind, self.features = "image", []
        self.rows: list[tuple[str, str, str, float]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.about = label("", "CardCaption", wrap=True)
        layout.addWidget(self.about)
        row = QHBoxLayout()
        self.pick = QPushButton(icon("folder-open"), "Choose…")
        self.pick.clicked.connect(self.choose)
        self.column = QComboBox()
        self.column.setMinimumWidth(140)
        self.go = QPushButton(icon("play"), "Test them all")
        self.go.setObjectName("Primary")
        self.go.clicked.connect(self.start)
        self.export = QPushButton(icon("download"), "Export CSV")
        self.export.clicked.connect(self.save)
        for widget in (self.pick, self.column, self.go, self.export):
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.source_label = PathLabel("")
        layout.addWidget(self.source_label)
        self.status = label("", "CardCaption", wrap=True)
        layout.addWidget(self.status)
        self.table = DataTable(("Item", "Predicted", "True", "Probability"), mono_columns=(3,),
                               stretch_column=0)
        layout.addWidget(self.table)
        self.source: Path | None = None
        self.configure(None, "image", [])

    def configure(self, run_dir: Path | None, kind: str, features: list[str]) -> None:
        self.run_dir, self.kind, self.features = run_dir, kind, features
        self.source = None
        self.source_label.setText("")
        files = kind in ("image", "audio")
        self.about.setText(("A folder of " + ("WAV files" if kind == "audio" else "images") +
                            "; files in sub-folders named after a class are scored against it.") if files else
                           "A CSV file: " + ("pick the column with the texts" if kind == "text" else
                                             f"columns {', '.join(features)}") +
                           "; a column named label, target or class is scored against.")
        self.column.setVisible(kind == "text")
        self.export.setEnabled(False)
        self.go.setEnabled(False)

    def choose(self) -> None:
        start = str(self.ctx.paths.workspace)
        if self.kind in ("image", "audio"):
            path = QFileDialog.getExistingDirectory(self, "Choose a folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a CSV file", start, "CSV files (*.csv)")
        if path:
            self.set_source(Path(path))

    def set_source(self, path: Path) -> None:
        self.source = path
        self.source_label.setText(str(path))
        self.column.clear()
        if path.is_file():
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                header = next(csv.reader(handle), [])
            self.column.addItems(header)
            texty = next((h for h in header if h.lower() in ("text", "message", "review", "sentence")), None)
            if texty:
                self.column.setCurrentText(texty)
        self.go.setEnabled(self.run_dir is not None)

    def start(self) -> None:
        if self.source is None or self.run_dir is None:
            return
        try:
            inputs, names, truth = collect(self.kind, self.source, self.column.currentText() or None,
                                           self.features)
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        if not inputs:
            self.status.setText("Nothing to test there.")
            return
        self.status.setText(f"Testing {len(inputs):,} items…")
        self.go.setEnabled(False)

        def done(result: dict[str, Any]) -> None:
            self.show_answers(names, truth, result.get("predictions", []))

        def failed(message: str) -> None:
            self.status.setText(f"Could not test them: {message}")
            self.go.setEnabled(True)

        def partial(data: dict[str, Any]) -> None:
            self.status.setText(f"{data.get('done', 0):,} of {data.get('total', 0):,} done…")

        self.send("predict", {"run": str(self.run_dir), "inputs": inputs}, partial, done, failed)

    def show_answers(self, names: list[str], truth: list[str], predictions: list[dict[str, Any]]) -> None:
        self.rows = []
        self.table.clear_rows()
        for name, true, answer in zip(names, truth, predictions, strict=False):
            if "value" in answer:
                predicted, share = f"{answer['value']:.4g}", 1.0
            else:
                predicted, share = answer["classes"][0], float(answer["probabilities"][0])
            self.rows.append((name, predicted, true, share))
            self.table.add_row((name, predicted, true or "—", f"{share:.3f}"))
        scored = [(p, t) for _n, p, t, _s in self.rows if t]
        accuracy = sum(p == t for p, t in scored) / len(scored) if scored else None
        self.status.setText(f"{len(self.rows):,} tested" + (f" · accuracy {accuracy:.1%} on {len(scored):,} "
                                                            "labelled"
                                                            if accuracy is not None else ""))
        self.export.setEnabled(bool(self.rows))
        self.go.setEnabled(True)

    def save(self) -> Path | None:
        if not self.rows:
            return None
        folder = self.ctx.paths.reports
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"batch-test-{datetime.now():%Y%m%d-%H%M%S}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item", "predicted", "true", "probability"])
            writer.writerows(self.rows)
        self.status.setText(f"Saved {target.name} in reports/")
        return target
